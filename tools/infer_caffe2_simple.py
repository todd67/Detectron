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

import detectron.utils.boxes as box_utils
from detectron.modeling.generate_anchors import generate_anchors

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
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def _create_cell_anchors(cfg):
    """
    Generate all types of anchors for all fpn levels/scales/aspect ratios.
    This function is called only once at the beginning of inference.
    """
    k_max, k_min = cfg.RPN_MAX_LEVEL, cfg.RPN_MIN_LEVEL
    scales_per_octave = cfg.SCALES_PER_OCTAVE
    aspect_ratios = cfg.ASPECT_RATIOS
    anchor_scale = cfg.ANCHOR_SCALE
    A = scales_per_octave * len(aspect_ratios)
    anchors = {}
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = np.zeros((A, 4))
        a = 0
        for octave in range(scales_per_octave):
            octave_scale = 2 ** (octave / float(scales_per_octave))
            for aspect in aspect_ratios:
                anchor_sizes = (stride * octave_scale * anchor_scale, )
                anchor_aspect_ratios = (aspect, )
                cell_anchors[a, :] = generate_anchors(
                    stride=stride, sizes=anchor_sizes,
                    aspect_ratios=anchor_aspect_ratios)
                a += 1
        anchors[lvl] = cell_anchors
    return anchors
    
    
def _retina_im_detect_box(cfg, im_shape, im_scale):
    """Generate RetinaNet detection boxes from workspace.
        return: boxes_all[cls] - list of detections per class
            Each class result is an np.array
                box[:, 0:4]: box coordinates
                box[:, 4]:   score   
    """

    # Although anchors are input independent and could be precomputed,
    # recomputing them per image only brings a small overhead
    anchors = _create_cell_anchors(cfg)

    cls_probs, box_preds = [], []
    k_max, k_min = cfg.RPN_MAX_LEVEL, cfg.RPN_MIN_LEVEL
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
        pred_boxes /= im_scale
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
    """ Post-processing of detection output.
        return: np.array - list of detections
            box[:, 0:4]: box coordinates
            box[:, 4]:   score   
            box[:, 5]:   class ID
    """

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

    return detections


def _prepare_blob(cfg, im):
    # Subtract mean
    im = im.astype(np.float32, copy=False)
    im -= np.array(cfg.PIXEL_MEANS)
    im_shape = [im.shape[0], im.shape[1]]

    # Resize
    im_scale = float(cfg.SCALE) / float(min(im_shape))
    if np.round(im_scale * max(im_shape)) > cfg.MAX_SIZE:
        im_scale = float(cfg.MAX_SIZE) / float(max(im_shape))
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
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


def run_retina_net(cfg, net, im):
    data, im_scale = _prepare_blob(cfg, im)
    workspace.FeedBlob('data', data)
            
    try:
        workspace.RunNet(net)
    except Exception as e:
        print('Running pb model failed.\n{}'.format(e))
        exit(-1)

    boxes_all = _retina_im_detect_box(cfg, im.shape, im_scale)
    cls_boxes = _detection_post_processing(cfg, boxes_all)

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

    return net, net_init


def prepare_workspace(net, net_init, optimize=False): 
    workspace.ResetWorkspace()
    workspace.RunNetOnce(net_init)

    if optimize: 
        print('Optimizing memory usage...')
        optim_proto = memonger.optimize_inference_for_dag(net, ["data"])
        net = core.Net(optim_proto)

    workspace.CreateNet(net, input_blobs=['data'])
    return net

def print_detections(results, thresh): 
    # Hard-coded class names for COCO dataset
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]    
    for ret in results: 
        score = ret[4]
        lbl   = classes[int(ret[5])]       
        box   = [int(x) for x in ret[:4]]

        if score >= thresh: 
            print('\tlabel={:15}  score={:.4f}  box={}'.format(lbl[:15], score, box))

def main(args):
    # Load cfg
    with open(os.path.join(args.model_dir, 'model.cfg'), 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
    # Load net to GPU
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, args.gpu_id)):
        net, net_init = load_model(args.model_dir)

    # Prepare workspace
    net = prepare_workspace(net, net_init, optimize=args.do_optimize)

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        print('Processing', im_name)
        im = cv2.imread(im_name)

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, args.gpu_id)):
            results = run_retina_net(cfg, net, im)

        print_detections(results, thresh=0.7)

        if i >= args.num: 
            break


if __name__ == '__main__':   
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
