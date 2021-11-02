import torch
import torch.nn as nn
from .bilinear_info import SparseBilinearInfo
from .utility_layers import SparseWeights2d, EmitEncoding, GradientBlock
import torch.nn.functional as F

class EstimatorOutputs(nn.Identity):
    """
    Gathers all of the outputs of the Bilinear Info Estimators in the model.
    """

    def __init__(self, *args, **kwargs):
        super(EstimatorOutputs, self).__init__(*args, **kwargs)
        self.estimator_outputs = []

    def append(self, x):
        self.estimator_outputs.append(x)
        return x

    def get_outputs(self):
        return self.estimator_outputs

    def clear_outputs(self):
        self.estimator_outputs = []

class EncodingOutputs(nn.Identity):
    """
    Gathers all of the outputs of the EmitEncoding layers in the model.
    """

    def __init__(self, *args, **kwargs):
        super(EncodingOutputs, self).__init__(*args, **kwargs)
        self.encoding_outputs = []

    def append(self, x):
        self.encoding_outputs.append(x)
        return x

    def get_outputs(self):
        return self.encoding_outputs

    def clear_outputs(self):
        self.encoding_outputs = []

class EstimatorBlock(nn.Module):
    def __init__(self,
                 estimator_outputs,
                 encoding_outputs,
                 in_channels,
                 sparsity,
                 negative_samples=16,
                 k_predictions=5,
                 sparse_weights_class=SparseWeights2d):
        super(EstimatorBlock, self).__init__()
        self.estimator_outputs = estimator_outputs
        self.encoding_outputs = encoding_outputs
        self.bilinear_info = SparseBilinearInfo(in_channels,
                                                in_channels,
                                                sparsity,
                                                negative_samples,
                                                k_predictions)
        self.emit_encoding = EmitEncoding(in_channels)
        self.gradient_block = GradientBlock()

    def forward_unsupervised(self, x, n_patches_x, n_patches_y):
        out = F.adaptive_avg_pool2d(x, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()
        info_estimate = self.bilinear_info.estimate_info(out, out)
        self.estimator_outputs.append(info_estimate)

    def forward_supervised(self, x, n_patches_x, n_patches_y):
        encoded = self.emit_encoding.encode(x, n_patches_x, n_patches_y)
        self.encoding_outputs.append(encoded)