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

"""Implements ShuffleNet v2.

See: https://arxiv.org/pdf/1807.11164.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict

from detectron.core.config import cfg

# ---------------------------------------------------------------------------- #
# Bits for specific architectures 
# ---------------------------------------------------------------------------- #

def add_ShuffleNetV2_half_body(model):
    return add_ShuffleNetV2_body(model, width=0.5)

# ---------------------------------------------------------------------------- #
# Generic SqueezeNext components
# ---------------------------------------------------------------------------- #

def add_ConvBN(model, in_name, out_name, in_dim, out_dim, kernel, **kwargs):     
    p = model.Conv(in_name, out_name, in_dim, out_dim, kernel, no_bias=1, **kwargs)
    return model.AffineChannel(p, out_name + '_bn', out_dim, inplace=True)

def add_ConvBNReLU(model, in_name, out_name, in_dim, out_dim, kernel, **kwargs): 
    p = add_ConvBN(model, in_name, out_name, in_dim, out_dim, kernel, **kwargs)
    return model.Relu(p, p)

def add_ShuffleBlockV2(model, in_name, out_name, in_dim, out_dim, stride):     
    out_dim //= 2

    # Shortcut
    if stride == 1:
        left  = out_name + '_slice1'
        right = out_name + '_slice2'
        model.net.Split(in_name, [left, right])
        in_dim //= 2
    else:
        left  = add_ConvBN(model, in_name, out_name + '_conv4', in_dim, in_dim, 3, pad=1, stride=stride, group=in_dim)
        left  = add_ConvBNReLU(model, left, out_name + '_conv5', in_dim, out_dim, 1)
        right = in_name

    # Main branch
    right = add_ConvBNReLU(model, right, out_name + '_conv1', in_dim, out_dim, 1)
    right = add_ConvBN(model, right, out_name + '_conv2', out_dim, out_dim, 3, pad=1, stride=stride, group=out_dim) # DepthwiseConv
    right = add_ConvBNReLU(model, right, out_name + '_conv3', out_dim, out_dim, 1)

    # Merge and shuffle
    p = model.Concat([left, right], out_name + '_concat', axis=1)  

    return model.net.ChannelShuffle(p, out_name + '_shuffle', group=2)

def add_ShuffleNetV2_body(model, width):
    width_config = {
        0.25: (24, 48, 96, 512),
        0.33: (32, 64, 128, 512),
        0.5: (48, 96, 192, 1024),
        1.0: (116, 232, 464, 1024),
        1.5: (176, 352, 704, 1024),
        2.0: (244, 488, 976, 2048),
    }
    width_config = width_config[width]
    blocks = [4, 8, 4] 

    p = model.AffineChannel('data', 'data_bn', 3, inplace=True)
    p = add_ConvBNReLU(model, p, 'stage1_conv', 3, 24, kernel=3, pad=1, stride=2)
    p = model.MaxPool(p, 'stage1_pool', kernel=3, pad=1, stride=2)

    in_dim = 24
    for stage, b in enumerate(blocks):         
        out_dim = width_config[stage]
        stage_name = 'stage_' + str(2 + stage)

        p = add_ShuffleBlockV2(model, p, stage_name + '_1', in_dim, out_dim, 2)

        for c in xrange(b-1): 
            name = stage_name + '_' + str(c+2)
            p = add_ShuffleBlockV2(model, p, name, out_dim, out_dim, 1)

        in_dim = out_dim

    # Last Conv
    out_dim = width_config[3]
    p = add_ConvBNReLU(model, p, 'conv5', in_dim, out_dim, kernel=1)

    return p, out_dim, 1. / 32.
