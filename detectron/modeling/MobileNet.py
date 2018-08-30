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

"""MobileNet from https://arxiv.org/abs/1704.04861."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg

def add_conv3x3_bn(model, in_name, out_name, in_dim, out_dim, stride):
    p = model.Conv(in_name, out_name, in_dim, out_dim, 3, pad=1, stride=stride, no_bias=1)
    p = model.AffineChannel(p, out_name + '_bn', out_dim, inplace=True)
    p = model.Relu(p, p)
    return p

def add_conv3x3_dw(model, in_name, out_name, in_dim, out_dim, stride):
    p = model.Conv(in_name, out_name + '_dw', in_dim, in_dim, 3, pad=1, stride=stride, group=in_dim, no_bias=1)
    p = model.AffineChannel(p, out_name + '_dw_bn', in_dim, inplace=True)
    p = model.Relu(p, p)

    p = model.Conv(p, out_name + '_sep', in_dim, out_dim, 1, pad=0, stride=1, no_bias=1)
    p = model.AffineChannel(p, out_name + '_sep_bn', out_dim, inplace=True)
    p = model.Relu(p, p)
    return p

def add_MobileNet_conv6_body(model):
    p = add_conv3x3_bn(model, 'data', 'conv1', 3, 32, stride=2)

    p = add_conv3x3_dw(model, p, 'conv2_1',   32,   64, stride=1)
    p = add_conv3x3_dw(model, p, 'conv2_2',   64,  128, stride=2)
    p = add_conv3x3_dw(model, p, 'conv3_1',  128,  128, stride=1)
    p = add_conv3x3_dw(model, p, 'conv3_2',  128,  256, stride=2)
    p = add_conv3x3_dw(model, p, 'conv4_1',  256,  256, stride=1)
    p = add_conv3x3_dw(model, p, 'conv4_2',  256,  512, stride=2)
    p = add_conv3x3_dw(model, p, 'conv5_1',  512,  512, stride=1)
    p = add_conv3x3_dw(model, p, 'conv5_2',  512,  512, stride=1)
    p = add_conv3x3_dw(model, p, 'conv5_3',  512,  512, stride=1)
    p = add_conv3x3_dw(model, p, 'conv5_4',  512,  512, stride=1)
    p = add_conv3x3_dw(model, p, 'conv5_5',  512,  512, stride=1)
    p = add_conv3x3_dw(model, p, 'conv5_6',  512, 1024, stride=2)
    p = add_conv3x3_dw(model, p, 'conv6'  , 1024, 1024, stride=1)

    return p, 1024, 1. / 32.

