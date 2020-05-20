import numpy as np
import torch
import torch.nn.functional as F

from nupic.torch.modules import SparseWeights, KWinners
from torch import nn


class DendriteInput(nn.Module):
    def __init__(self,
                 in_dim,
                 n_dendrites,
                 threshold=2,
                 sparse_weights=True,
                 weight_sparsity=0.2,
                 percent_on=0.1,
                 k_inference_factor=1,
                 boost_strength=2.,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 ):
        super(DendriteInput, self).__init__()
        self.threshold = threshold
        linear = nn.Linear(in_dim, n_dendrites)

        if sparse_weights:
            self.linear = SparseWeights(linear, weight_sparsity)
        else:
            self.linear = linear

        if 0 < percent_on < 1.0:
            self.act_fun = KWinners(
                n=n_dendrites,
                percent_on=percent_on,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )
        else:
            self.act_fun = nn.ReLU()

        # self.act_fun = ADA_fun().cuda()

    def dendrite_activation(self, x):
        return torch.clamp(x,min=self.threshold)
        # return torch.max(torch.Tensor([0.]).to(torch.device(self.device_type)),
        #  x - self.threshold)

    def forward(self, x):
        out = self.linear(x)
        # return ADA_fun(out)
        return self.act_fun(out)


class DendriteOutput(nn.Module):
    def __init__(self, out_dim, dpc):
        super(DendriteOutput, self).__init__()
        self.dpc = dpc
        self.register_buffer("mask", self.dend_mask(out_dim))
        self.weight = torch.nn.Parameter(torch.Tensor(out_dim, dpc * out_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(out_dim))
        # nn.init.kaiming_uniform_(self.weight)
        # self.bias.data.fill_(0.)

    def forward(self, x):
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)

    def dend_mask(self, out_dim):
        mask = torch.zeros(out_dim, out_dim)
        torch.diag
        inds = np.diag_indices(out_dim)
        mask[inds[0], inds[1]] = 1.
        out_mask = torch.repeat_interleave(mask, self.dpc, dim=0).T
        return out_mask


class DendriteLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 dpc,
                 threshold=2,
                 sparse_weights=True,
                 weight_sparsity=0.2,
                 percent_on=0.1,
                 ):
        super(DendriteLayer, self).__init__()

        self.dpc = dpc
        self.n_dendrites = out_dim * self.dpc
        self.threshold = threshold
        self.input = DendriteInput(
            in_dim=in_dim,
            n_dendrites=self.n_dendrites,
            threshold=self.threshold,
            sparse_weights=sparse_weights,
            weight_sparsity=weight_sparsity,
            percent_on=percent_on,
        )
        self.output = DendriteOutput(out_dim, self.dpc)

    def forward(self, x):
        out1 = self.input(x)
        out2 = self.output(out1)
        return out2


# class ADA_fun(nn.Module):
#     def __init__(self, a=1, c=1, l=0.005):
#         super(ADA_fun, self).__init__()
#         self.a = a
#         self.c = c
#         self.l = l

#     def forward(self, x):
#         neg_relu = torch.clamp(x, max=0)
#         ADA = F.relu(x) * torch.exp(-x * self.a + self.c)
#         ADA_l = self.l * neg_relu + ADA
#         return ADA_l
