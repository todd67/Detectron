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

"""Implements Thin ResNet.

See: https://github.com/jay-mahadeokar/pynetbuilder/tree/master/models/imagenet.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.net import get_group_gn


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def add_ResNet50_1by2_body(model):
    return add_ResNet_body(model, 
        blocks = (3, 4, 6, 3), 
        is_half = True
    )

# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #

def add_ConvBN(model, in_name, out_name, in_dim, out_dim, kernel, pad, stride): 
    p = model.Conv(in_name, 'conv_' + out_name, in_dim, out_dim, kernel, pad=pad, stride=stride)
    return model.AffineChannel(p, 'bn_' + out_name, out_dim, inplace=True)

def add_ConvBNReLU(model, in_name, out_name, in_dim, out_dim, kernel, pad, stride): 
    p = add_ConvBN(model, in_name, out_name, in_dim, out_dim, kernel, pad, stride)
    return model.Relu(p, p)

def add_EltwiseRelu(model, in_names, out_name): 
    p = model.net.Sum(in_names, 'eltwise_' + out_name)
    return model.Relu(p, p)

def add_Shortcut(model, in_name, out_name, in_dim, out_dim, shortcut, stride):
    # Shortcut branch
    if shortcut == 'identity':  
        shortcut_p   = in_name

    elif shortcut == 'projection': 
        name = out_name + '_proj_shortcut'
        shortcut_p   = add_ConvBN(model, in_name, name, in_dim, out_dim, kernel=1, pad=0, stride=stride)

    # Main branch - bottleneck
    p = in_name
    p = add_ConvBNReLU(model, p, out_name + '_branch2a', in_dim,      out_dim // 4,  kernel=1, pad=0, stride=stride)
    p = add_ConvBNReLU(model, p, out_name + '_branch2b', out_dim // 4, out_dim // 4, kernel=3, pad=1, stride=1)
    p = add_ConvBN    (model, p, out_name + '_branch2c', out_dim // 4, out_dim,          kernel=1, pad=0, stride=1)

    # Combine
    return add_EltwiseRelu(model, [shortcut_p, p], out_name)

def add_ResNet_body(model, blocks, num_output_stage1=64, is_half=False): 
    # Stage 1
    p = add_ConvBNReLU(model, 'data', '1', 3, num_output_stage1, kernel=7, pad=3, stride=2)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)    

    if is_half: 
        base_num_output = num_output_stage1 * 2
    else: 
        base_num_output = num_output_stage1 * 4

    prev_num_output = num_output_stage1

    for stage in xrange(4): 
        curr_num_output = base_num_output * (2 ** stage)

        for block in range(blocks[stage]): 
            if block == 0: 
                shortcut = 'projection'
                stride = 2 if stage > 0 else 1
            else: 
                shortcut = 'identity'
                stride = 1

            name = 'stage{}_block{}'.format(stage, block)

            p = add_Shortcut(model, p, name, prev_num_output, curr_num_output,
                shortcut=shortcut, stride=stride) 

            prev_num_output = curr_num_output

    return p, prev_num_output, 1. / 32.