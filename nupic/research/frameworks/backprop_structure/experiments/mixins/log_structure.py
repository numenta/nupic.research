# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import torch

from nupic.research.frameworks.backprop_structure.modules import (
    BinaryGatedConv2d,
    BinaryGatedLinear,
    HardConcreteGatedConv2d,
    HardConcreteGatedLinear,
)


def nonzero_counts(model):
    result = {}

    for layername, layer in model.named_modules():
        if isinstance(layer, (BinaryGatedConv2d,
                              BinaryGatedLinear,
                              HardConcreteGatedConv2d,
                              HardConcreteGatedLinear)):
            num_inputs = 1.
            for d in layer.weight_size()[1:]:
                num_inputs *= d

            i_nz_by_unit = layer.get_inference_nonzeros()

            multiplies, adds = layer.count_inference_flops()

            result[layername] = {
                "hist_inference_nz_by_unit": i_nz_by_unit.tolist(),
                "inference_nz": torch.sum(i_nz_by_unit).item(),
                "num_input_units": num_inputs,
                "multiplies": multiplies,
                "flops": multiplies + adds,
            }

    inference_nz = 0
    flops = 0
    multiplies = 0
    for layername in result.keys():
        inference_nz += result[layername]["inference_nz"]
        flops += result[layername]["flops"]
        multiplies += result[layername]["multiplies"]

    result["inference_nz"] = inference_nz
    result["flops"] = flops
    result["multiplies"] = multiplies

    return result


class LogStructure(object):
    def run_epoch(self, iteration):
        result = super().run_epoch(iteration)
        result.update(nonzero_counts(self.model))
        return result
