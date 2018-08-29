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

"""VGG_CNN_M_1024 from https://arxiv.org/abs/1405.3531."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg


def add_VGG_CNN_M_1024_conv5_body(model):
    model.Conv('data', 'conv1', 3, 96, 7, pad=3, stride=2)
    model.Relu('conv1', 'conv1')
    model.LRN('conv1', 'norm1', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm1', 'pool1', kernel=3, pad=1, stride=2)
    model.StopGradient('pool1', 'pool1')
    # No updates at conv1 and below (norm1 and pool1 have no params,
    # so we can stop gradients before them, too)
    model.Conv('pool1', 'conv2', 96, 256, 5, pad=2, stride=2)
    model.Relu('conv2', 'conv2')
    model.LRN('conv2', 'norm2', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm2', 'pool2', kernel=3, pad=1, stride=2)
    model.Conv('pool2', 'conv3', 256, 512, 3, pad=1, stride=1)
    model.Relu('conv3', 'conv3')
    model.Conv('conv3', 'conv4', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4', 'conv4')
    model.Conv('conv4', 'conv5', 512, 512, 3, pad=1, stride=1)
    blob_out = model.Relu('conv5', 'conv5')
    return blob_out, 512, 1. / 16.

def add_VGG_CNN_M_1024_conv5_pool_body(model):
    add_VGG_CNN_M_1024_conv5_body(model)
    blob_out = model.MaxPool('conv5', 'pool5', kernel=3, pad=1, stride=2)
    return blob_out, 512, 1. / 32.

def add_VGG_CNN_M_1024_conv6_body(model):
    add_VGG_CNN_M_1024_conv5_pool_body(model)
    model.Conv('pool5', 'conv6', 512, 1024, 3, pad=1, stride=1)
    blob_out = model.Relu('conv6', 'conv6')
    return blob_out, 1024, 1. / 32.

def add_VGG_CNN_M_1024_conv8_body(model): 
    add_VGG_CNN_M_1024_conv5_body(model)
    model.MaxPool('conv5', 'pool5', kernel=3, pad=1, stride=1)
    model.Conv('pool5', 'fc6_conv', 512, 512, kernel=3, pad=6, dilation=6, stride=1)
    model.Relu('fc6_conv', 'fc6_conv')
    model.Conv('fc6_conv', 'fc7_conv', 512, 512, 1, stride=1)
    model.Conv('fc7_conv', 'conv8', 512, 1024, 2, stride = 2)
    blob_out = model.Relu('conv8', 'conv8')
    return blob_out, 1024, 1. / 32.

def add_VGG_CNN_M_1024_roi_fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', 4096, 1024)
    blob_out = model.Relu('fc7', 'fc7')
    return blob_out, 1024
