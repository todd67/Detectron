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

"""VGG16 from https://arxiv.org/abs/1409.1556."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg


def add_VGG16_conv5_body(model):
    model.Conv('data', 'conv1_1', 3, 64, 3, pad=1, stride=1)
    model.Relu('conv1_1', 'conv1_1')
    model.Conv('conv1_1', 'conv1_2', 64, 64, 3, pad=1, stride=1)
    model.Relu('conv1_2', 'conv1_2')
    model.MaxPool('conv1_2', 'pool1', kernel=2, pad=0, stride=2)
    model.Conv('pool1', 'conv2_1', 64, 128, 3, pad=1, stride=1)
    model.Relu('conv2_1', 'conv2_1')
    model.Conv('conv2_1', 'conv2_2', 128, 128, 3, pad=1, stride=1)
    model.Relu('conv2_2', 'conv2_2')
    model.MaxPool('conv2_2', 'pool2', kernel=2, pad=0, stride=2)
    model.StopGradient('pool2', 'pool2')
    model.Conv('pool2', 'conv3_1', 128, 256, 3, pad=1, stride=1)
    model.Relu('conv3_1', 'conv3_1')
    model.Conv('conv3_1', 'conv3_2', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_2', 'conv3_2')
    model.Conv('conv3_2', 'conv3_3', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_3', 'conv3_3')
    model.MaxPool('conv3_3', 'pool3', kernel=2, pad=0, stride=2)
    model.Conv('pool3', 'conv4_1', 256, 512, 3, pad=1, stride=1)
    model.Relu('conv4_1', 'conv4_1')
    model.Conv('conv4_1', 'conv4_2', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_2', 'conv4_2')
    model.Conv('conv4_2', 'conv4_3', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_3', 'conv4_3')
    model.MaxPool('conv4_3', 'pool4', kernel=2, pad=0, stride=2)
    model.Conv('pool4', 'conv5_1', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv5_1', 'conv5_1')
    model.Conv('conv5_1', 'conv5_2', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv5_2', 'conv5_2')
    model.Conv('conv5_2', 'conv5_3', 512, 512, 3, pad=1, stride=1)
    blob_out = model.Relu('conv5_3', 'conv5_3')
    return blob_out, 512, 1. / 16.

def add_VGG16_conv5_pool_body(model):
    add_VGG16_conv5_body(model)
    blob_out = model.MaxPool('conv5_3', 'pool5', kernel=2, pad=0, stride=2)
    return blob_out, 1024, 1. / 32.

def add_VGG16_conv6_body(model):
    add_VGG16_conv5_pool_body(model)
    model.Conv('pool5', 'conv6_1', 512, 1024, 3, pad=1, stride=1)
    model.Relu('conv6_1', 'conv6_1')
    model.Conv('conv6_1', 'conv6_2', 1024, 1024, 3, pad=1, stride=1)
    blob_out = model.Relu('conv6_2', 'conv6_2')
    return blob_out, 1024, 1. / 32.

def add_VGG16_conv8_body(model): 
    add_VGG16_conv5_body(model)
    model.MaxPool('conv5_3', 'pool5', kernel = 3, pad = 1, stride = 1)
    model.Conv('pool5', 'fc6_conv', 512, 1024, 3, pad = 6, dilation = 6, stride = 1)
    model.Relu('fc6_conv', 'fc6_conv')
    model.Conv('fc6_conv', 'fc7_conv', 1024, 1024, 1)
    model.Relu('fc7_conv', 'fc7_conv')
    model.Conv('fc7_conv', 'conv8', 1024, 2048, 2, stride = 2)
    blob_out = model.Relu('conv8', 'conv8')
    return blob_out, 2048, 1. / 32.  

def add_VGG16_roi_fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC('pool5', 'fc6', dim_in * 7 * 7, 4096)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', 4096, 4096)
    blob_out = model.Relu('fc7', 'fc7')
    return blob_out, 4096
