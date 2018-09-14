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

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.

Using purely caffe2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import os
import sys
import logging
import numpy as np
import yaml

from caffe2.python import core, workspace, memonger
from caffe2.proto import caffe2_pb2
import google.protobuf.text_format

from detectron.utils.timer import Timer
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.vis as vis_utils
import detectron.utils.boxes as box_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        help='path to exported model',
        required=True,
        type=str
    )
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        help='select GPU',
        default=0,
        type=int
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--optimize',
        dest='do_optimize',
        help='optimize execution to reduce memory usage',
        action='store_true'
    )
    parser.add_argument(
        '--num',
        dest='num',
        help='maximum number of images to run',
        default=0, 
        type=int
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()
 
    
def _retina_im_detect_box(cfg, im_shape, im_scale):
    """Generate RetinaNet detection boxes from workspace.
        return: boxes_all[cls] - list of detections per class
            Each class result is an np.array
                box[:, 0:4]: box coordinates
                box[:, 4]:   score   
    """
    cls_probs, box_preds = [], []
    k_min, k_max = min(cfg.ANCHORS.keys()), max(cfg.ANCHORS.keys())-1    
    A = cfg.SCALES_PER_OCTAVE * len(cfg.ASPECT_RATIOS)
    
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_probs.append(core.ScopedName('retnet_cls_prob_{}'.format(suffix)))
        box_preds.append(core.ScopedName('retnet_bbox_pred_{}'.format(suffix)))

    cls_probs = workspace.FetchBlobs(cls_probs)
    box_preds = workspace.FetchBlobs(box_preds)

    # here the boxes_all are [x0, y0, x1, y1, score]
    boxes_all = [[] for _ in xrange(cfg.NUM_CLASSES)]

    cnt = 0
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = cfg.ANCHORS[lvl]

        # fetch per level probability
        cls_prob = cls_probs[cnt]
        box_pred = box_preds[cnt]
        cls_prob = cls_prob.reshape((
            cls_prob.shape[0], A, int(cls_prob.shape[1] / A),
            cls_prob.shape[2], cls_prob.shape[3]))
        box_pred = box_pred.reshape((
            box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]))
        cnt += 1

        if cfg.SOFTMAX:
            cls_prob = cls_prob[:, :, 1::, :, :]

        cls_prob_ravel = cls_prob.ravel()
        # In some cases [especially for very small img sizes], it's possible that
        # candidate_ind is empty if we impose threshold 0.05 at all levels. This
        # will lead to errors since no detections are found for this image. Hence,
        # for lvl 7 which has small spatial resolution, we take the threshold 0.0
        th = cfg.INFERENCE_TH if lvl < k_max else 0.0
        candidate_inds = np.where(cls_prob_ravel > th)[0]
        if (len(candidate_inds) == 0):
            continue

        pre_nms_topn = min(cfg.PRE_NMS_TOP_N, len(candidate_inds))
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

        if not cfg.CLASS_SPECIFIC_BBOX:
            box_deltas = box_pred[0, anchor_ids, :, y, x]
        else:
            box_cls_inds = classes * 4
            box_deltas = np.vstack(
                [box_pred[0, ind:ind + 4, yi, xi]
                 for ind, yi, xi in zip(box_cls_inds, y, x)]
            )
        pred_boxes = (
            box_utils.bbox_transform(boxes, box_deltas)
            if cfg.BBOX_REG else boxes)
        pred_boxes /= np.array([[im_scale[1], im_scale[0], im_scale[1], im_scale[0]]])
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im_shape)
        box_scores = np.zeros((pred_boxes.shape[0], 5))
        box_scores[:, 0:4] = pred_boxes
        box_scores[:, 4] = scores

        for cls in range(1, cfg.NUM_CLASSES):
            inds = np.where(classes == cls - 1)[0]
            if len(inds) > 0:
                boxes_all[cls].extend(box_scores[inds, :])

    return boxes_all


def _detection_post_processing(cfg, boxes_all):
    """Post-processing of detection output."""

    # Combine predictions across all levels and retain the top scoring by class
    detections = []
    for cls, boxes in enumerate(boxes_all):
        if not boxes: 
            continue

        cls_dets = np.vstack(boxes).astype(dtype=np.float32)
        # do class specific nms here
        keep = box_utils.nms(cls_dets, cfg.NMS)
        cls_dets = cls_dets[keep, :]
        out = np.zeros((len(keep), 6))
        out[:, 0:5] = cls_dets
        out[:, 5].fill(cls)
        detections.append(out)

    detections = np.vstack(detections)
    inds = np.argsort(-detections[:, 4])
    detections = detections[inds[0:cfg.DETECTIONS_PER_IM], :]

    # Convert the detections to image cls_ format 
    num_classes = cfg.NUM_CLASSES
    cls_boxes = [[] for _ in range(cfg.NUM_CLASSES)]
    for c in range(1, num_classes):
        inds = np.where(detections[:, 5] == c)[0]
        cls_boxes[c] = detections[inds, :5]

    return cls_boxes


def _prepare_blob(cfg, im):
    # Subtract mean
    im = im.astype(np.float32, copy=False)
    im -= np.array(cfg.PIXEL_MEANS)
    im_shape = [im.shape[0], im.shape[1]]

    # Resize
    if cfg.SQUASH: 
        im_scale = float(cfg.SCALE) / np.array(im_shape[0:2], dtype=np.float32)
    else: 
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(cfg.SCALE) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > cfg.MAX_SIZE:
            im_scale = float(cfg.MAX_SIZE) / float(im_size_max)
        im_scale = np.array([im_scale, im_scale])

    im = cv2.resize(im, None, None, fx=im_scale[1], fy=im_scale[0],
                    interpolation=cv2.INTER_LINEAR)

    # Pad the image so they can be divisible by a stride
    stride = float(cfg.COARSEST_STRIDE)
    shape = [int(np.ceil(im.shape[0] / stride) * stride),
             int(np.ceil(im.shape[1] / stride) * stride)]

    blob = np.zeros((1, shape[0], shape[1], 3), dtype=np.float32)
    blob[0, 0:im.shape[0], 0:im.shape[1], :] = im

    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    return blob, im_scale


def run_retina_net(cfg, net, im, timers):
    timers['prepare_data'].tic()
    data, im_scale = _prepare_blob(cfg, im)
    workspace.FeedBlob('data', data)
    timers['prepare_data'].toc()
            
    timers['run_net'].tic()
    try:
        workspace.RunNet(net)
    except Exception as e:
        print('Running pb model failed.\n{}'.format(e))
        exit(-1)
    timers['run_net'].toc()

    timers['im_detect_box'].tic()
    boxes_all = _retina_im_detect_box(cfg, im.shape, im_scale)
    timers['im_detect_box'].toc()

    timers['post_processing'].tic()
    cls_boxes = _detection_post_processing(cfg, boxes_all)
    timers['post_processing'].toc()

    return cls_boxes


def load_model(model_dir): 
    netdef = caffe2_pb2.NetDef()
    with open(os.path.join(model_dir, 'model.pbtxt'), 'r') as f: 
        netdef = google.protobuf.text_format.Merge(str(f.read()), netdef)

    net = core.Net(netdef)

    netdef = caffe2_pb2.NetDef()
    with open(os.path.join(model_dir, 'model_init.pb'), 'rb') as f: 
        netdef.ParseFromString(f.read())

    net_init = core.Net(netdef)

    # Load model configurations
    with open(os.path.join(model_dir, 'model.cfg'), 'r') as f:
        cfg = AttrDict(yaml.load(f))

    cfg.ANCHORS = {k: np.array(v) for k, v in cfg.ANCHORS.iteritems()}

    return cfg, net, net_init


def prepare_workspace(net, net_init, optimize=False): 
    workspace.ResetWorkspace()
    workspace.RunNetOnce(net_init)

    if optimize: 
        print('Optimizing memory usage...')
        optim_proto = memonger.optimize_inference_for_dag(net, ["data"])
        net = core.Net(optim_proto)

    workspace.CreateNet(net, input_blobs=['data'])
    return net

def main(args):
    # Load net to GPU
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, args.gpu_id)):
        cfg, net, net_init = load_model(args.model_dir)

    # Prepare workspace
    net = prepare_workspace(net, net_init, optimize=args.do_optimize)

    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    WARM_UP = 3

    total_timer = Timer()
    timers = defaultdict(Timer)

    for i, im_name in enumerate(im_list):
        if args.output_dir: 
            out_name = os.path.join(
                args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
            )
        print('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)

        total_timer.tic()
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, args.gpu_id)):
            cls_boxes = run_retina_net(
                cfg, net, im, timers=timers
            )
        total_timer.toc()
        print('Total time: {:.3f}s'.format(total_timer.diff))
        for k, v in timers.items():
            print(' | {}: {:.3f}s'.format(k, v.average_time))

        if i >= args.num: 
            break

        if i == WARM_UP: 
            total_timer.reset()
            for k, v in timers.items(): 
                v.reset()

        if args.output_dir: 
            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                im_name,
                args.output_dir,
                cls_boxes,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2,
                ext=args.output_ext,
                out_when_no_box=True
            )

    print('Average: ')
    print('Total time: {:.3f}s'.format(total_timer.average_time))

    for k, v in timers.items():
        print(' | {}: {:.3f}s'.format(k, v.average_time))

if __name__ == '__main__':   
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
