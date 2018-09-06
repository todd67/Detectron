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

"""Implements SqueezeNet.

See: https://arxiv.org/abs/1707.01083.
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

def add_ShuffleNet_1x_g3_body(model):
    return add_ShuffleNet_body(model, width=1.0, group=3)

# ---------------------------------------------------------------------------- #
# Generic SqueezeNext components
# ---------------------------------------------------------------------------- #

def add_ConvBN(model, in_name, out_name, in_dim, out_dim, kernel, **kwargs): 
    p = model.Conv(in_name, out_name, in_dim, out_dim, kernel, no_bias=1, **kwargs)
    return model.AffineChannel(p, out_name + '_bn', out_dim, inplace=True)

def add_ConvBNReLU(model, in_name, out_name, in_dim, out_dim, kernel, **kwargs): 
    p = add_ConvBN(model, in_name, out_name, in_dim, out_dim, kernel, **kwargs)
    return model.Relu(p, p)

def add_ShuffleBlock(model, counters, in_name, in_dim, out_dim, stride, group): 
    is_shuffle = (counters['block_id'] > 0)  # Skip shuffle for first block    
    cur_group  = group if is_shuffle else 1

    counters['block_id'] += 1; 
    name = 'resx' + str(counters['block_id']); 

    # Shortcut
    if stride == 1: 
        shortcut_p = in_name

    else: 
        out_name = name + '_match_conv'
        shortcut_p = model.AveragePool(in_name, out_name, kernel=3, pad=1, stride=stride)
        out_dim -= in_dim

    # Main branch
    squeeze_dim = out_dim // 4
    p = add_ConvBNReLU(model, in_name, name + '_conv1', in_dim, squeeze_dim, 1, pad=0, stride=1, group=cur_group)

    if is_shuffle: 
        p = model.net.ChannelShuffle(p, 'shuffle' + str(counters['block_id']), group=group)

    p = add_ConvBN(model, p, name + '_conv2', squeeze_dim, squeeze_dim, 3, pad=1, stride=stride, group=squeeze_dim) # DepthwiseConv
    p = add_ConvBN(model, p, name + '_conv3', squeeze_dim, out_dim,     1, pad=0, stride=1,      group=group)

    # Combine
    if stride == 1: 
        p = model.Sum([shortcut_p, p], name + '_elewise') 
    else: 
        p = model.Concat([shortcut_p, p], name + '_concat', axis=1)  

    return model.Relu(p, p)

def add_ShuffleNet_body(model, width, group):
    stage2_width = [-1, 144, 200, 240, 272, 384]
    blocks = [4, 8, 4] 

    p = add_ConvBNReLU(model, 'data', 'conv1', 3, 24, kernel=3, pad=1, stride=2)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)    

    counters = defaultdict(int)
    in_dim = 24
    for stage, b in enumerate(blocks): 
        out_dim = int(stage2_width[group] * width) * (2 ** stage)

        for c in xrange(b): 
            if c == 0: 
                stride = 2
            else: 
                stride = 1

            p = add_ShuffleBlock(model, counters, p, in_dim, out_dim, stride, group)
            in_dim = out_dim

    return p, in_dim, 1. / 32.
