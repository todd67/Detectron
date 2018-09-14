#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Script to convert the model (.yaml and .pkl) trained by train_net to a
standard Caffe2 model in pb format (model.pb and model_init.pb). The converted
model is good for production usage, as it could run independently and efficiently
on CPU, GPU and mobile without depending on the detectron codebase.

Please see Caffe2 tutorial (
https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html) for loading
the converted model, and run_model_pb() for running the model for inference.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import os
import pprint
import sys
import yaml

from collections import defaultdict

import caffe2.python.utils as putils
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.modeling import generate_anchors
from detectron.utils.logging import setup_logging
from detectron.utils.collections import AttrDict
from detectron.utils.model_convert_utils import convert_op_in_proto
from detectron.utils.model_convert_utils import op_filter
from detectron.core.test_retinanet import _create_cell_anchors
import detectron.utils.boxes as box_utils
import detectron.utils.blob as blob_utils
import detectron.core.test_engine as test_engine
import detectron.utils.c2 as c2_utils
import detectron.utils.model_convert_utils as mutils
import detectron.utils.vis as vis_utils

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logger = setup_logging(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a trained network to pb format'
    )
    parser.add_argument(
        '--cfg', dest='cfg_file', help='optional config file', default=None,
        type=str)
    parser.add_argument(
        '--net_name', dest='net_name', help='optional name for the net',
        default="detectron", type=str)
    parser.add_argument(
        '--out_dir', dest='out_dir', help='output dir', default=None,
        type=str)
    parser.add_argument(
        '--test_img', dest='test_img',
        help='optional test image, used to verify the model conversion',
        default=None,
        type=str)
    parser.add_argument(
        '--fuse_af', dest='fuse_af', help='1 to fuse_af',
        default=1,
        type=int)
    parser.add_argument(
        '--device', dest='device',
        help='Device to run the model on',
        choices=['cpu', 'gpu'],
        default='cpu',
        type=str)
    parser.add_argument(
        '--net_execution_type', dest='net_execution_type',
        help='caffe2 net execution type',
        choices=['simple', 'dag'],
        default='simple',
        type=str)
    parser.add_argument(
        '--use_nnpack', dest='use_nnpack',
        help='Use nnpack for conv',
        default=1,
        type=int)
    parser.add_argument(
        'opts', help='See detectron/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    ret = parser.parse_args()
    ret.out_dir = os.path.abspath(ret.out_dir)
    if ret.device == 'gpu' and ret.use_nnpack:
        logger.warn('Should not use mobile engine for gpu model.')
        ret.use_nnpack = 0

    return ret


def unscope_name(name):
    return c2_utils.UnscopeName(name)


def reset_names(names):
    for i in range(len(names)):
        names[i] = unscope_name(names[i])


def convert_collect_and_distribute(
    op, blobs,
    roi_canonical_scale,
    roi_canonical_level,
    roi_max_level,
    roi_min_level,
    rpn_max_level,
    rpn_min_level,
    rpn_post_nms_topN,
):
    print('Converting CollectAndDistributeFpnRpnProposals'
          ' Python -> C++:\n{}'.format(op))
    assert op.name.startswith('CollectAndDistributeFpnRpnProposalsOp'), \
        'Not valid CollectAndDistributeFpnRpnProposalsOp'

    inputs = [x for x in op.input]
    ret = core.CreateOperator(
        'CollectAndDistributeFpnRpnProposals',
        inputs,
        list(op.output),
        roi_canonical_scale=roi_canonical_scale,
        roi_canonical_level=roi_canonical_level,
        roi_max_level=roi_max_level,
        roi_min_level=roi_min_level,
        rpn_max_level=rpn_max_level,
        rpn_min_level=rpn_min_level,
        rpn_post_nms_topN=rpn_post_nms_topN,
    )
    return ret


def convert_gen_proposals(
    op, blobs,
    rpn_pre_nms_topN,
    rpn_post_nms_topN,
    rpn_nms_thresh,
    rpn_min_size,
):
    print('Converting GenerateProposals Python -> C++:\n{}'.format(op))
    assert op.name.startswith('GenerateProposalsOp'), 'Not valid GenerateProposalsOp'

    spatial_scale = mutils.get_op_arg_valf(op, 'spatial_scale', None)
    assert spatial_scale is not None

    lvl = int(op.input[0][-1]) if op.input[0][-1].isdigit() else None

    inputs = [x for x in op.input]
    anchor_name = 'anchor{}'.format(lvl) if lvl else 'anchor'
    inputs.append(anchor_name)
    anchor_sizes = (cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - cfg.FPN.RPN_MIN_LEVEL),) if lvl else cfg.RPN.SIZES
    blobs[anchor_name] = get_anchors(spatial_scale, anchor_sizes)
    print('anchors {}'.format(blobs[anchor_name]))

    ret = core.CreateOperator(
        'GenerateProposals',
        inputs,
        list(op.output),
        spatial_scale=spatial_scale,
        pre_nms_topN=rpn_pre_nms_topN,
        post_nms_topN=rpn_post_nms_topN,
        nms_thresh=rpn_nms_thresh,
        min_size=rpn_min_size,
        correct_transform_coords=True,
    )
    return ret, anchor_name


def get_anchors(spatial_scale, anchor_sizes):
    anchors = generate_anchors.generate_anchors(
        stride=1. / spatial_scale,
        sizes=anchor_sizes,
        aspect_ratios=cfg.RPN.ASPECT_RATIOS).astype(np.float32)
    return anchors


def reset_blob_names(blobs):
    ret = {unscope_name(x): blobs[x] for x in blobs}
    blobs.clear()
    blobs.update(ret)


def convert_net(args, net, blobs):

    @op_filter()
    def convert_op_name(op):
        if args.device != 'gpu':
            if op.engine != 'DEPTHWISE_3x3':
                op.engine = ''
            op.device_option.CopyFrom(caffe2_pb2.DeviceOption())
        reset_names(op.input)
        reset_names(op.output)
        return [op]

    @op_filter(type='Python')
    def convert_python(op):
        if op.name.startswith('GenerateProposalsOp'):
            gen_proposals_op, ext_input = convert_gen_proposals(
                op, blobs,
                rpn_min_size=float(cfg.TEST.RPN_MIN_SIZE),
                rpn_post_nms_topN=cfg.TEST.RPN_POST_NMS_TOP_N,
                rpn_pre_nms_topN=cfg.TEST.RPN_PRE_NMS_TOP_N,
                rpn_nms_thresh=cfg.TEST.RPN_NMS_THRESH,
            )
            net.external_input.extend([ext_input])
            return [gen_proposals_op]
        elif op.name.startswith('CollectAndDistributeFpnRpnProposalsOp'):
            collect_dist_op = convert_collect_and_distribute(
                op, blobs,
                roi_canonical_scale=cfg.FPN.ROI_CANONICAL_SCALE,
                roi_canonical_level=cfg.FPN.ROI_CANONICAL_LEVEL,
                roi_max_level=cfg.FPN.ROI_MAX_LEVEL,
                roi_min_level=cfg.FPN.ROI_MIN_LEVEL,
                rpn_max_level=cfg.FPN.RPN_MAX_LEVEL,
                rpn_min_level=cfg.FPN.RPN_MIN_LEVEL,
                rpn_post_nms_topN=cfg.TEST.RPN_POST_NMS_TOP_N,
            )
            return [collect_dist_op]
        else:
            raise ValueError('Failed to convert Python op {}'.format(
                op.name))

    # Only convert UpsampleNearest to ResizeNearest when converting to pb so that the existing models is unchanged
    # https://github.com/facebookresearch/Detectron/pull/372#issuecomment-410248561
    @op_filter(type='UpsampleNearest')
    def convert_upsample_nearest(op):
        for arg in op.arg:
            if arg.name == 'scale':
                scale = arg.i
                break
        else:
            raise KeyError('No attribute "scale" in UpsampleNearest op')
        resize_nearest_op = core.CreateOperator('ResizeNearest',
                                                list(op.input),
                                                list(op.output),
                                                name=op.name,
                                                width_scale=float(scale),
                                                height_scale=float(scale))
        return resize_nearest_op

    @op_filter()
    def convert_rpn_rois(op):
        for j in range(len(op.input)):
            if op.input[j] == 'rois':
                print('Converting op {} input name: rois -> rpn_rois:\n{}'.format(
                    op.type, op))
                op.input[j] = 'rpn_rois'
        for j in range(len(op.output)):
            if op.output[j] == 'rois':
                print('Converting op {} output name: rois -> rpn_rois:\n{}'.format(
                    op.type, op))
                op.output[j] = 'rpn_rois'
        return [op]

    @op_filter(type_in=['StopGradient', 'Alias'])
    def convert_remove_op(op):
        print('Removing op {}:\n{}'.format(op.type, op))
        return []

    # We want to apply to all operators, including converted
    # so run separately
    convert_op_in_proto(net, convert_remove_op)
    convert_op_in_proto(net, convert_upsample_nearest)
    convert_op_in_proto(net, convert_python)
    convert_op_in_proto(net, convert_op_name)
    convert_op_in_proto(net, convert_rpn_rois)

    reset_names(net.external_input)
    reset_names(net.external_output)

    reset_blob_names(blobs)


def add_bbox_ops(args, net, blobs):
    new_ops = []
    new_external_outputs = []

    # Operators for bboxes
    op_box = core.CreateOperator(
        "BBoxTransform",
        ['rpn_rois', 'bbox_pred', 'im_info'],
        ['pred_bbox'],
        weights=cfg.MODEL.BBOX_REG_WEIGHTS,
        apply_scale=False,
        correct_transform_coords=True,
    )
    new_ops.extend([op_box])

    blob_prob = 'cls_prob'
    blob_box = 'pred_bbox'
    op_nms = core.CreateOperator(
        "BoxWithNMSLimit",
        [blob_prob, blob_box],
        ['score_nms', 'bbox_nms', 'class_nms'],
        arg=[
            putils.MakeArgument("score_thresh", cfg.TEST.SCORE_THRESH),
            putils.MakeArgument("nms", cfg.TEST.NMS),
            putils.MakeArgument("detections_per_im", cfg.TEST.DETECTIONS_PER_IM),
            putils.MakeArgument("soft_nms_enabled", cfg.TEST.SOFT_NMS.ENABLED),
            putils.MakeArgument("soft_nms_method", cfg.TEST.SOFT_NMS.METHOD),
            putils.MakeArgument("soft_nms_sigma", cfg.TEST.SOFT_NMS.SIGMA),
        ]
    )
    new_ops.extend([op_nms])
    new_external_outputs.extend(['score_nms', 'bbox_nms', 'class_nms'])

    net.Proto().op.extend(new_ops)
    net.Proto().external_output.extend(new_external_outputs)


def add_retina_output(net):
    cls_probs, box_preds = get_retina_output_blob_names()

    net.Proto().external_output.extend(cls_probs)
    net.Proto().external_output.extend(box_preds)


def convert_model_gpu(args, net, init_net):
    assert args.device == 'gpu'

    ret_net = copy.deepcopy(net)
    ret_init_net = copy.deepcopy(init_net)

    cdo_cuda = mutils.get_device_option_cuda()
    cdo_cpu = mutils.get_device_option_cpu()

    CPU_OPS = [
        ["CollectAndDistributeFpnRpnProposals", None],
        ["GenerateProposals", None],
        ["BBoxTransform", None],
        ["BoxWithNMSLimit", None],
    ]
    CPU_BLOBS = ["im_info", "anchor"]

    @op_filter()
    def convert_op_gpu(op):
        for x in CPU_OPS:
            if mutils.filter_op(op, type=x[0], inputs=x[1]):
                return None
        op.device_option.CopyFrom(cdo_cuda)
        return [op]

    @op_filter()
    def convert_init_op_gpu(op):
        if op.output[0] in CPU_BLOBS:
            op.device_option.CopyFrom(cdo_cpu)
        else:
            op.device_option.CopyFrom(cdo_cuda)
        return [op]

    convert_op_in_proto(ret_init_net.Proto(), convert_init_op_gpu)
    convert_op_in_proto(ret_net.Proto(), convert_op_gpu)

    with open('net.pbtxt', 'w') as f: 
        f.write(str(ret_net.Proto()))

    with open('init_net.pbtxt',  'w') as f: 
        f.write(str(ret_init_net.Proto()))

    ret = core.InjectDeviceCopiesAmongNets([ret_init_net, ret_net])

    return [ret[0][1], ret[0][0]]


def gen_init_net(net, blobs, empty_blobs):
    blobs = copy.deepcopy(blobs)
    for x in empty_blobs:
        blobs[x] = np.array([], dtype=np.float32)
    init_net = mutils.gen_init_net_from_blobs(
        blobs, net.external_inputs)
    init_net = core.Net(init_net)
    return init_net


def _save_image_graphs(args, all_net, all_init_net):
    print('Saving model graph...')
    mutils.save_graph(
        all_net.Proto(), os.path.join(args.out_dir, "model_def.png"),
        op_only=False)
    print('Model def image saved to {}.'.format(args.out_dir))


def _save_models(all_net, all_init_net, args):
    print('Writing converted model to {}...'.format(args.out_dir))
    fname = "model"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.out_dir, fname + '.pb'), 'w') as f:
        f.write(all_net.Proto().SerializeToString())
    with open(os.path.join(args.out_dir, fname + '.pbtxt'), 'w') as f:
        f.write(str(all_net.Proto()))
    with open(os.path.join(args.out_dir, fname + '_init.pb'), 'w') as f:
        f.write(all_init_net.Proto().SerializeToString())

    # Also save inference CFG for RetinaNet
    if cfg.RETINANET.RETINANET_ON: 
        out_cfg = {
            'SCALES_PER_OCTAVE':    cfg.RETINANET.SCALES_PER_OCTAVE,
            'ASPECT_RATIOS':        list(cfg.RETINANET.ASPECT_RATIOS), 
            'SOFTMAX':              cfg.RETINANET.SOFTMAX, 
            'INFERENCE_TH':         cfg.RETINANET.INFERENCE_TH,
            'PRE_NMS_TOP_N':        cfg.RETINANET.PRE_NMS_TOP_N,
            'CLASS_SPECIFIC_BBOX':  cfg.RETINANET.CLASS_SPECIFIC_BBOX,
            'BBOX_REG':             cfg.TEST.BBOX_REG,
            'NUM_CLASSES':          cfg.MODEL.NUM_CLASSES,
            'NMS':                  cfg.TEST.NMS,
            'DETECTIONS_PER_IM':    cfg.TEST.DETECTIONS_PER_IM,
            'SCALE':                cfg.TEST.SCALE, 
            'MAX_SIZE':             cfg.TEST.MAX_SIZE, 
            'PIXEL_MEANS':          cfg.PIXEL_MEANS.tolist(),
            'COARSEST_STRIDE':      cfg.FPN.COARSEST_STRIDE,
            'ANCHORS':              {k: v.tolist() for k, v in _create_cell_anchors().iteritems()}
        }

    yaml.add_representer(unicode, lambda dumper, data: dumper.represent_str(data.encode('utf-8')))
    with open(os.path.join(args.out_dir, fname + '.cfg'), 'w') as f: 
        yaml.dump(out_cfg, f)

#    _save_image_graphs(args, all_net, all_init_net)


def load_model(args):
    model = test_engine.initialize_model_from_cfg(cfg.TEST.WEIGHTS)
    blobs = mutils.get_ws_blobs()

    return model, blobs


def _get_result_blobs(check_blobs):
    ret = {}
    for x in check_blobs:
        sn = core.ScopedName(x)
        if workspace.HasBlob(sn):
            ret[x] = workspace.FetchBlob(sn)
        else:
            ret[x] = None

    return ret


def _sort_results(boxes, segms, keypoints, classes):
    indices = np.argsort(boxes[:, -1])[::-1]
    if boxes is not None:
        boxes = boxes[indices, :]
    if segms is not None:
        segms = [segms[x] for x in indices]
    if keypoints is not None:
        keypoints = [keypoints[x] for x in indices]
    if classes is not None:
        if isinstance(classes, list):
            classes = [classes[x] for x in indices]
        else:
            classes = classes[indices]

    return boxes, segms, keypoints, classes


def run_model_cfg(args, im, check_blobs):
    workspace.ResetWorkspace()
    model, _ = load_model(args)
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = test_engine.im_detect_all(
            model, im, None, None,
        )

    boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
        cls_boxes, cls_segms, cls_keyps)

    # sort the results based on score for comparision
    boxes, segms, keypoints, classes = _sort_results(
        boxes, segms, keypoints, classes)

    # write final results back to workspace
    def _ornone(res):
        return np.array(res) if res is not None else np.array([], dtype=np.float32)
    with c2_utils.NamedCudaScope(0):
        workspace.FeedBlob(core.ScopedName('result_boxes'), _ornone(boxes))
        workspace.FeedBlob(core.ScopedName('result_segms'), _ornone(segms))
        workspace.FeedBlob(core.ScopedName('result_keypoints'), _ornone(keypoints))
        workspace.FeedBlob(core.ScopedName('result_classids'), _ornone(classes))

    # get result blobs
    with c2_utils.NamedCudaScope(0):
        ret = _get_result_blobs(check_blobs)

    return ret


def _prepare_blobs(
    im,
    pixel_means,
    target_size,
    max_size,
):
    ''' Reference: blob.prep_im_for_blob() '''

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape

    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    # Reuse code in blob_utils and fit FPN
    blob = blob_utils.im_list_to_blob([im])

    blobs = {}
    blobs['data'] = blob
    blobs['im_info'] = np.array(
        [[blob.shape[2], blob.shape[3], im_scale]],
        dtype=np.float32
    )
    return blobs


def get_retina_output_blob_names(): 
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL

    cls_probs, box_preds = [], []
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_probs.append(core.ScopedName('retnet_cls_prob_{}'.format(suffix)))
        box_preds.append(core.ScopedName('retnet_bbox_pred_{}'.format(suffix)))

    return cls_probs, box_preds


def retina_detection_output(im):
    """Generate RetinaNet detections on a single image."""

    # Although anchors are input independent and could be precomputed,
    # recomputing them per image only brings a small overhead
    anchors = _create_cell_anchors()

    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    A = cfg.RETINANET.SCALES_PER_OCTAVE * len(cfg.RETINANET.ASPECT_RATIOS)
    
    cls_probs, box_preds = get_retina_output_blob_names()

    cls_probs = workspace.FetchBlobs(cls_probs)
    box_preds = workspace.FetchBlobs(box_preds)
    im_info   = workspace.FetchBlob('im_info')
    im_scale  = im_info[0, 2]
    im_shape  = im.shape #(int(im_info[0, 1]), int(im_info[0, 0]))

    # here the boxes_all are [x0, y0, x1, y1, score]
    boxes_all = defaultdict(list)

    cnt = 0
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = anchors[lvl]

        # fetch per level probability
        cls_prob = cls_probs[cnt]
        box_pred = box_preds[cnt]
        cls_prob = cls_prob.reshape((
            cls_prob.shape[0], A, int(cls_prob.shape[1] / A),
            cls_prob.shape[2], cls_prob.shape[3]))
        box_pred = box_pred.reshape((
            box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]))
        cnt += 1

        if cfg.RETINANET.SOFTMAX:
            cls_prob = cls_prob[:, :, 1::, :, :]

        cls_prob_ravel = cls_prob.ravel()
        # In some cases [especially for very small img sizes], it's possible that
        # candidate_ind is empty if we impose threshold 0.05 at all levels. This
        # will lead to errors since no detections are found for this image. Hence,
        # for lvl 7 which has small spatial resolution, we take the threshold 0.0
        th = cfg.RETINANET.INFERENCE_TH if lvl < k_max else 0.0
        candidate_inds = np.where(cls_prob_ravel > th)[0]
        if (len(candidate_inds) == 0):
            continue

        pre_nms_topn = min(cfg.RETINANET.PRE_NMS_TOP_N, len(candidate_inds))
        inds = np.argpartition(
            cls_prob_ravel[candidate_inds], -pre_nms_topn)[-pre_nms_topn:]
        inds = candidate_inds[inds]

        inds_5d = np.array(np.unravel_index(inds, cls_prob.shape)).transpose()
        classes = inds_5d[:, 2]
        anchor_ids, y, x = inds_5d[:, 1], inds_5d[:, 3], inds_5d[:, 4]
        scores = cls_prob[:, anchor_ids, classes, y, x]

        boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
        boxes *= stride
        boxes += cell_anchors[anchor_ids, :]

        if not cfg.RETINANET.CLASS_SPECIFIC_BBOX:
            box_deltas = box_pred[0, anchor_ids, :, y, x]
        else:
            box_cls_inds = classes * 4
            box_deltas = np.vstack(
                [box_pred[0, ind:ind + 4, yi, xi]
                 for ind, yi, xi in zip(box_cls_inds, y, x)]
            )
        pred_boxes = (
            box_utils.bbox_transform(boxes, box_deltas)
            if cfg.TEST.BBOX_REG else boxes)
        pred_boxes /= im_scale
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im_shape)
        box_scores = np.zeros((pred_boxes.shape[0], 5))
        box_scores[:, 0:4] = pred_boxes
        box_scores[:, 4] = scores

        for cls in range(1, cfg.MODEL.NUM_CLASSES):
            inds = np.where(classes == cls - 1)[0]
            if len(inds) > 0:
                boxes_all[cls].extend(box_scores[inds, :])

    # Combine predictions across all levels and retain the top scoring by class
    detections = []
    for cls, boxes in boxes_all.items():
        cls_dets = np.vstack(boxes).astype(dtype=np.float32)
        # do class specific nms here
        keep = box_utils.nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        out = np.zeros((len(keep), 6))
        out[:, 0:5] = cls_dets
        out[:, 5].fill(cls)
        detections.append(out)

    detections = np.vstack(detections)
    inds = np.argsort(-detections[:, 4])
    detections = detections[inds[0:cfg.TEST.DETECTIONS_PER_IM], :]

    # detections (N, 6) format:
    #   detections[:, :4] - boxes
    #   detections[:, 4] - scores
    #   detections[:, 5] - classes
    bboxes  = detections[:, :5]
    classes = detections[:, 5]

    return classes, bboxes

def run_model_pb(args, net, init_net, im, check_blobs):
    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)
    mutils.create_input_blobs_for_net(net.Proto())
    workspace.CreateNet(net)

    # input_blobs, _ = core_test._get_blobs(im, None)
    input_blobs = _prepare_blobs(
        im,
        cfg.PIXEL_MEANS,
        cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
    )
    gpu_blobs = []
    if args.device == 'gpu':
        gpu_blobs = ['data']
    for k, v in input_blobs.items():
        workspace.FeedBlob(
            core.ScopedName(k),
            v,
            mutils.get_device_option_cuda() if k in gpu_blobs else
            mutils.get_device_option_cpu()
        )

    try:
        workspace.RunNet(net)

        if not cfg.RETINANET.RETINANET_ON:
            scores = workspace.FetchBlob('score_nms')
            classids = workspace.FetchBlob('class_nms')
            boxes = workspace.FetchBlob('bbox_nms')

            boxes = np.column_stack((boxes, scores))
        else: 
            classids, boxes = retina_detection_output(im)

    except Exception as e:
        print('Running pb model failed.\n{}'.format(e))
        # may not detect anything at all
        R = 0
        boxes = np.zeros((R, 5), dtype=np.float32)
        classids = np.zeros((R,), dtype=np.float32)

    # sort the results based on score for comparision
    boxes, _, _, classids = _sort_results(
        boxes, None, None, classids)

    # write final result back to workspace
    workspace.FeedBlob('result_boxes', boxes)
    workspace.FeedBlob('result_classids', classids)

    ret = _get_result_blobs(check_blobs)

    return ret


def verify_model(args, model_pb, test_img_file):
    check_blobs = [
        "result_boxes", "result_classids",  # result
    ]

    print('Loading test file {}...'.format(test_img_file))
    test_img = cv2.imread(test_img_file)
    assert test_img is not None

    def _run_cfg_func(im, blobs):
        return run_model_cfg(args, im, check_blobs)

    def _run_pb_func(im, blobs):
        return run_model_pb(args, model_pb[0], model_pb[1], im, check_blobs)

    print('Checking models...')
    assert mutils.compare_model(
        _run_cfg_func, _run_pb_func, test_img, check_blobs)


def main():
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    logger.info('Converting model with config:')
    logger.info(pprint.pformat(cfg))

    # script will stop when it can't find an operator rather
    # than stopping based on these flags
    #
    # assert not cfg.MODEL.KEYPOINTS_ON, "Keypoint model not supported."
    # assert not cfg.MODEL.MASK_ON, "Mask model not supported."
    # assert not cfg.FPN.FPN_ON, "FPN not supported."
    # assert not cfg.RETINANET.RETINANET_ON, "RetinaNet model not supported."

    # load model from cfg
    model, blobs = load_model(args)

    net = core.Net('')
    net.Proto().op.extend(copy.deepcopy(model.net.Proto().op))
    net.Proto().external_input.extend(
        copy.deepcopy(model.net.Proto().external_input))
    net.Proto().external_output.extend(
        copy.deepcopy(model.net.Proto().external_output))
    net.Proto().type = args.net_execution_type
    net.Proto().num_workers = 1 if args.net_execution_type == 'simple' else 4

    # Reset the device_option, change to unscope name and replace python operators
    convert_net(args, net.Proto(), blobs)

    # add operators for bbox
    if not cfg.RETINANET.RETINANET_ON: 
        add_bbox_ops(args, net, blobs)
        empty_blobs = ['data', 'im_info']
    else: 
        add_retina_output(net)
        logger.info('For RetinaNet, we do not add bbox processing ops.')
        empty_blobs = ['data']

    if args.fuse_af:
        print('Fusing affine channel...')
        net, blobs = mutils.fuse_net_affine(
            net, blobs)

    if args.use_nnpack:
        mutils.update_mobile_engines(net.Proto())

    # generate init net
    init_net = gen_init_net(net, blobs, empty_blobs)

    if args.device == 'gpu':
        [net, init_net] = convert_model_gpu(args, net, init_net)

    net.Proto().name = args.net_name
    init_net.Proto().name = args.net_name + "_init"

    if args.test_img is not None:
        verify_model(args, [net, init_net], args.test_img)

    _save_models(net, init_net, args)


if __name__ == '__main__':
    main()
