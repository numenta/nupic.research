import torch
import torch.nn as nn
from .bilinear_info import SparseBilinearInfo, BilinearInfo
from .utility_layers import SparseWeights2d, EmitEncoding, GradientBlock
import torch.nn.functional as F

class InfoEstimateAggregator(nn.Identity):
    """
    Aggregates all of the outputs of the Bilinear Info Estimators in the model into a single list.
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
    Gathers all of the outputs of the EmitEncoding layers in the model.
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
                 estimator_outputs,
                 encoding_outputs,
                 in_channels,
                 negative_samples=16,
                 k_predictions=5):
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
        self.estimator_outputs = estimator_outputs
        self.encoding_outputs = encoding_outputs
        self.bilinear_info = BilinearInfo(in_channels,
                                          in_channels,
                                          negative_samples,
                                          k_predictions)
        self.emit_encoding = EmitEncoding(in_channels)
        self.gradient_block = GradientBlock()

    """
    During unsupervised training, this function will be linked to the forward hook 
    for its corresponding module.
    """
    def forward_unsupervised(self, x, n_patches_x, n_patches_y):
        out = F.adaptive_avg_pool2d(x, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()
        info_estimate = self.bilinear_info.estimate_info(out, out)
        self.estimator_outputs.append(info_estimate)

    """
    During supervised training, this function will be linked to the forward hook for
    its corresponding module.
    """
    def forward_supervised(self, x, n_patches_x, n_patches_y):
        encoded = self.emit_encoding.encode(x, n_patches_x, n_patches_y)
        self.encoding_outputs.append(encoded)

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