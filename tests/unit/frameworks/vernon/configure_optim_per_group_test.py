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
# ----------------------------------------------------------------------

import unittest

from torch.nn.modules.batchnorm import _BatchNorm

from nupic.research.frameworks.pytorch.models import resnets
from nupic.research.frameworks.vernon.mixins import ConfigureOptimizerParamGroups

create_optimizer = ConfigureOptimizerParamGroups.create_optimizer
config = dict(
    optimizer_args=dict(lr=0.01, weight_decay=0.1),
    optim_args_groups=[
        # Group 0: Stem - slower lr
        dict(
            group_args=dict(lr=0.001),
            include_names=["features.stem.weight"],
        ),
        # Group 1: Batchnorm and Bias - no weight decay
        dict(
            group_args=dict(weight_decay=0),
            include_patterns=[".*bias"],
            include_modules=[_BatchNorm],
        ),
    ]
)


class ConfigureOptimizerParamGroupsTest(unittest.TestCase):

    def test_create_optimizer_for_resnet(self):
        """
        Filter out only the linear params of resnet.
        """
        resnet = resnets.resnet50(num_classes=10)

        optim = create_optimizer(config, resnet)
        self.assertEqual(len(optim.param_groups), 3)

        # Group 0 - reduced lr on stem
        lr = optim.param_groups[0]["lr"]
        num_params = len(optim.param_groups[0]["params"])
        weight_decay = optim.param_groups[0]["weight_decay"]
        self.assertEqual(lr, 0.001)
        self.assertEqual(num_params, 1)
        self.assertEqual(weight_decay, 0.1)

        # Group 1 - no weight decay on batch norm and bias params
        lr = optim.param_groups[1]["lr"]
        num_params = len(optim.param_groups[1]["params"])
        weight_decay = optim.param_groups[1]["weight_decay"]
        self.assertEqual(lr, 0.01)
        self.assertEqual(num_params, 107)
        self.assertEqual(weight_decay, 0)

        # Group 3: The remaining params; used default optim args.
        lr = optim.param_groups[2]["lr"]
        weight_decay = optim.param_groups[2]["weight_decay"]
        self.assertEqual(lr, 0.01)
        self.assertEqual(weight_decay, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
