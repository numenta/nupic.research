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

from nupic.research.frameworks.backprop_structure.modules import (
    BinaryGatedConv2d,
    BinaryGatedLinear,
    HardConcreteGatedConv2d,
    HardConcreteGatedLinear,
)


def constrain_parameters(module):
    if isinstance(module, (BinaryGatedConv2d,
                           BinaryGatedLinear,
                           HardConcreteGatedConv2d,
                           HardConcreteGatedLinear)):
        module.constrain_parameters()


class ConstrainParameters(object):
    def _after_optimizer_step(self):
        super()._after_optimizer_step()
        self.network.apply(constrain_parameters)
