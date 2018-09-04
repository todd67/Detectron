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

"""Implements SqueezeNext.

See: https://arxiv.org/pdf/1803.10615.pdf.
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

def add_SqNxt23_body(model):
    return add_SqNxt_body(model, blocks = (6, 6, 8, 1))

# ---------------------------------------------------------------------------- #
# Generic SqueezeNext components
# ---------------------------------------------------------------------------- #

def add_ConvBNReLU(model, counters, in_name, in_dim, out_dim, kernel, **kwargs): 
    counters['conv'] += 1; name = 'Convolution' + str(counters['conv']); 
    p = model.Conv(in_name, name, in_dim, out_dim, kernel, **kwargs)
    counters['bn'] += 1;  name = 'BatchNorm' + str(counters['bn']); 
    p = model.AffineChannel(p, name, out_dim, inplace=True)
    return model.Relu(p, p)

def add_EltwiseRelu(model, counters, in_names): 
    counters['eltwise'] += 1; name = 'Eltwise' + str(counters['eltwise'])
    p = model.net.Sum(in_names, name)
    return model.Relu(p, p)

def add_SqBlock(model, counters, in_name, in_dim, out_dim, shortcut, stride):
    # Shortcut branch
    if shortcut == 'projection': 
        shortcut_p = add_ConvBNReLU(model, counters, in_name, in_dim, out_dim, kernel=1, pad=0, stride=stride)

    else: # shortcut == 'identity'
        shortcut_p = in_name

    # Main branch
    p = in_name

    # Bottleneck    
    p = add_ConvBNReLU(model, counters, p, in_dim,       out_dim // 2, kernel=1, pad=0, stride=stride)
    p = add_ConvBNReLU(model, counters, p, out_dim // 2, out_dim // 4, kernel=1, pad=0, stride=1)

    # Separable filter
    counters['sqblock'] += 1

    if counters['sqblock'] % 2 == 1: 
        p = add_ConvBNReLU(model, counters, p, out_dim // 4, out_dim // 2, kernel=[3, 1], pads=[1, 0, 1, 0], stride=1)
        p = add_ConvBNReLU(model, counters, p, out_dim // 2, out_dim // 2, kernel=[1, 3], pads=[0, 1, 0, 1], stride=1)
    else: 
        p = add_ConvBNReLU(model, counters, p, out_dim // 4, out_dim // 2, kernel=[1, 3], pads=[0, 1, 0, 1], stride=1)
        p = add_ConvBNReLU(model, counters, p, out_dim // 2, out_dim // 2, kernel=[3, 1], pads=[1, 0, 1, 0], stride=1)

    # Expansion
    p = add_ConvBNReLU(model, counters, p, out_dim // 2, out_dim,      kernel=1, pad=0, stride=1)
        
    # Combine
    return add_EltwiseRelu(model, counters, [shortcut_p, p])

def add_SqNxt_body(model, blocks): 
    counters = defaultdict(int); 

    freeze_at = cfg.TRAIN.FREEZE_AT
    assert freeze_at in [0, 2, 3, 4, 5]

    # Stage 1
    p = add_ConvBNReLU(model, counters, 'data', 3, 64, kernel=7, pad=3, stride=2)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)    

    prev_num_output = 64
    base_num_output = 64

    for stage in xrange(4): 
        curr_num_output = base_num_output * (2 ** stage)

        for block in range(blocks[stage]): 
            if block == 0: 
                shortcut = 'projection'
                stride = 2 if stage > 0 else 1
            else: 
                shortcut = 'identity'
                stride = 1
            
            p = add_SqBlock(model, counters, p, prev_num_output, curr_num_output,
                shortcut=shortcut, stride=stride) 

            prev_num_output = curr_num_output

        if freeze_at == stage + 2: 
            model.StopGradient(p, p)

    # Final conv
    p = add_ConvBNReLU(model, counters, p, prev_num_output, 128, kernel=1, pad=0, stride=1)

    return p, 128, 1. / 32.