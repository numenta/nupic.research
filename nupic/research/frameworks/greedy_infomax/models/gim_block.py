# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
#
# This work was built off the Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The original Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
from .bilinear_info import SparseBilinearInfo, BilinearInfo
from .utility_layers import SparseWeights2d, EmitEncoding, GradientBlock
import torch.nn.functional as F

class InfoEstimateAggregator(nn.Identity):
    """
    Aggregates all the outputs of the Bilinear Info Estimators in the model into
    a single list.
    """

    def __init__(self, *args, **kwargs):
        super(InfoEstimateAggregator, self).__init__(*args, **kwargs)
        self.info_estimates = []

    def append(self, x):
        self.info_estimates.append(x)
        return x

    def get_outputs(self):
        return self.info_estimates

    def clear_outputs(self):
        self.info_estimates = []

class EncodingAggregator(nn.Identity):
    """
    Gathers all of the outputs of the EmitEncoding layers in the model into a single
    list.
    """

    def __init__(self, *args, **kwargs):
        super(EncodingAggregator, self).__init__(*args, **kwargs)
        self.encodings = []

    def append(self, x):
        self.encodings.append(x)
        return x

    def get_outputs(self):
        return self.encodings

    def clear_outputs(self):
        self.encodings = []

class GreedyInfoMaxBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 estimate_info_aggregator,
                 encoding_aggregator,
                 negative_samples=16,
                 k_predictions=5,
                 n_patches_x=None,
                 n_patches_y=None,):
        """
        A block that can be placed after any module in a model which consists of:
        1. A BilinearInfo module
        2. An EmitEncoding module
        3. A GradientBlock module

        In GreedyInfoMax experiments, this block represents the segregation of the
        gradients between the modules that come before and the modules that come
        after this block. These can be placed after any module in a model. Note that
        it is not completely necessary that the gradients be blocked, or even that
        the EmitEncoding blocks to be placed after the BilinearInfo block, but this
        is the most common use case.
        """
        super(GreedyInfoMaxBlock, self).__init__()
        self.info_estimate_aggregator = estimate_info_aggregator
        self.encoding_aggregator = encoding_aggregator
        self.negative_samples = negative_samples
        self.k_predictions = k_predictions
        self.in_channels = in_channels
        self.bilinear_info = BilinearInfo(in_channels,
                                          in_channels,
                                          self.negative_samples,
                                          self.k_predictions)
        self.emit_encoding = EmitEncoding(in_channels)
        self.gradient_block = GradientBlock()
        self.n_patches_x, self.n_patches_y = n_patches_x, n_patches_y


    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = out.reshape(-1, self.n_patches_x, self.n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()
        info_estimate = self.bilinear_info.estimate_info(out, out)
        self.info_estimate_aggregator.append(info_estimate)
        x_blocked = self.gradient_block(x).clone()
        return x_blocked

    """
    During unsupervised training, this function will be linked to the forward hook 
    for its corresponding module.
    """
    def wrapped_forward(self, module, input, output):
        return self.forward(output)


    def encode(self, x):
        encoded = self.emit_encoding.encode(x, self.n_patches_x, self.n_patches_y)
        self.encoding_aggregator.append(encoded)

    """
    During supervised training, this function will be linked to the forward hook for
    its corresponding module.
    """
    def wrapped_encode(self, module, input, output):
        return self.encode(output)

class SparseGreedyInfoMaxBlock(GreedyInfoMaxBlock):
    """
    A version of the above GreedyInfoMaxBlock which uses SparseBilinearInfo instead
    of a regular BilinearInfo module.
    """
    def __init__(self,
                 estimator_outputs,
                 encoding_outputs,
                 in_channels,
                 negative_samples=16,
                 k_predictions=5,
                 sparsity=None):
        super(SparseGreedyInfoMaxBlock, self).__init__(estimator_outputs,
                                                       encoding_outputs,
                                                       in_channels,
                                                       negative_samples,
                                                       k_predictions)
        self.sparsity = sparsity
        self.bilinear_info = SparseBilinearInfo(in_channels,
                                                in_channels,
                                                negative_samples,
                                                k_predictions,
                                                sparsity)