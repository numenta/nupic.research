# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
# ----------------------------------------------------------------------

from torch import nn

from nupic.research.frameworks.backprop_structure.modules import (
    VDropConv2d,
    VDropLinear,
)
from nupic.torch.modules import SparseWeights, SparseWeights2d


def maskedvdrop_to_sparseweights(from_model, to_model):
    # Load batchnorm. This will also load Conv/Linear biases in cases where the
    # from_module has a bias and the to_module isn't sparse.
    to_model.load_state_dict(from_model.state_dict(), strict=False)

    vdrop_data = from_model.vdrop_data
    z_mu = (vdrop_data.z_mu
            * vdrop_data.z_mask
            * (vdrop_data.compute_z_logalpha() < vdrop_data.threshold)).float()
    z_mask = vdrop_data.z_mask

    for from_module, w_mu, w_mask in zip(vdrop_data.modules,
                                         z_mu.split(vdrop_data.z_chunk_sizes),
                                         z_mask.split(vdrop_data.z_chunk_sizes)):
        name = [name
                for name, m in from_model.named_modules()
                if m is from_module][0]
        to_module = [m
                     for name2, m in to_model.named_modules()
                     if name2 == name][0]

        if ((isinstance(from_module, VDropLinear)
             and isinstance(to_module, nn.Linear))
            or (isinstance(from_module, VDropConv2d)
                and isinstance(to_module, nn.Conv2d))):
            to_module.weight.data[:] = w_mu.view(to_module.weight.shape)
            # Bias is covered by load_state_dict
        elif ((isinstance(from_module, VDropLinear)
               and isinstance(to_module, SparseWeights))
              or (isinstance(from_module, VDropConv2d)
                  and isinstance(to_module, SparseWeights2d))):
            w_mu = w_mu.view(to_module.module.weight.shape)
            to_module.module.weight.data[:] = w_mu

            w_mask = w_mask.view(to_module.zero_mask.shape)
            to_module.zero_mask.zero_()
            to_module.zero_mask[w_mask == 0.0] = 1.0

            if from_module.bias is not None:
                to_module.module.bias.data[:] = from_module.bias

            to_module.rezero_weights()
        else:
            raise ValueError("This function can't convert from "
                             f"{from_module} to {to_module}")
