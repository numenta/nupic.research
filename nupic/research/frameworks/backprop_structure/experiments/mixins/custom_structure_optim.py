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

from nupic.research.frameworks.backprop_structure.modules import (
    BinaryGatedConv2d,
    BinaryGatedLinear,
)


class CustomStructureOptim(object):
    def __init__(self, structure_optim_params, **kwargs):
        self.structure_optim_params = structure_optim_params
        super().__init__(**kwargs)

    def _get_parameters(self):
        main_parameters = []
        structure_parameters = []
        for i, module in enumerate(self.network.modules()):
            # Skip the nn.Sequential module
            if i > 0:
                if isinstance(module, (BinaryGatedConv2d,
                                       BinaryGatedLinear)):
                    main_parameters += [module.exc_weight, module.inh_weight]
                    if module.use_bias:
                        main_parameters.append(module.bias)
                    structure_parameters += [module.exc_p1, module.inh_p1]
                else:
                    main_parameters += list(module.parameters())

        return [{"params": main_parameters},
                {"params": structure_parameters,
                 **self.structure_optim_params}]
