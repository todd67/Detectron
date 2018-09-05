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

See: https://arxiv.org/abs/1602.07360.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict

from detectron.core.config import cfg

# ---------------------------------------------------------------------------- #
# Generic SqueezeNext components
# ---------------------------------------------------------------------------- #

def add_ConvReLU(model, in_name, out_name, in_dim, out_dim, kernel, **kwargs): 
    p = model.Conv(in_name, out_name, in_dim, out_dim, kernel, **kwargs)
    return model.Relu(p, p)

def add_FireBlock(model, in_name, prefix, in_dim, out_dim):        
    p = in_name
    p = add_ConvReLU(model, p, prefix + '_squeeze1x1', in_dim,       out_dim // 4, kernel=1, pad=0)
    e1 = add_ConvReLU(model, p, prefix + '_expand1x1', out_dim // 4, out_dim,      kernel=1, pad=0)
    e2 = add_ConvReLU(model, p, prefix + '_expand3x3', out_dim // 4, out_dim,      kernel=3, pad=1)
    return model.Concat([e1, e2], prefix + '_concat', axis=1)  

def add_SqNet1_1_body(model): 
    p = add_ConvReLU(model, 'data', 'conv1', 3, 64, kernel=3, pad=1, stride=2)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)    

    p = add_FireBlock(model, p, 'fire2',  64, 64)
    p = add_FireBlock(model, p, 'fire3',  128, 64)
    p = model.MaxPool(p, 'pool3', kernel=3, pad=1, stride=2)    

    p = add_FireBlock(model, p, 'fire4', 128, 128)
    p = add_FireBlock(model, p, 'fire5', 256, 128)
    p = model.MaxPool(p, 'pool5', kernel=3, pad=1, stride=2)    

    p = add_FireBlock(model, p, 'fire6', 256, 192)
    p = add_FireBlock(model, p, 'fire7', 384, 192)
    p = add_FireBlock(model, p, 'fire8', 384, 256)
    p = add_FireBlock(model, p, 'fire9', 512, 256)

    p = add_ConvReLU(model, p, 'fire10_conv', 512, 1024, kernel=2, stride=2)

    return p, 1024, 1. / 32.
