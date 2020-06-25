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


class Regularize(object):
    def __init__(self, reg_schedule=None, downscale_reg_with_training_set=False,
                 **kwargs):
        """
        @param reg_schedule (dict)
        Mapping from epoch number to the reg_weight to use on that timestep and
        afterward.

        @param downscale_reg_with_training_set (bool)
        If True, multiply the regularization term by (1 / size_of_training_set)
        """
        super().__init__(**kwargs)

        if downscale_reg_with_training_set:
            self.reg_coefficient = 1 / len(self.dataset_manager.get_train_dataset(0))
        else:
            self.reg_coefficient = 1

        if reg_schedule is None:
            self.reg_schedule = {}
            self.reg_weight = 1.0
        else:
            self.reg_schedule = reg_schedule
            self.reg_weight = reg_schedule[0]

    def _regularization(self):
        reg = None  # Perform accumulation on the device.
        for layer in self.network.modules():
            if hasattr(layer, "regularization"):
                if reg is None:
                    reg = layer.regularization()
                else:
                    reg += layer.regularization()

        if reg is None:
            return 0
        else:
            return (self.reg_weight
                    * self.reg_coefficient
                    * reg)

    def run_epoch(self, iteration):
        if iteration in self.reg_schedule:
            self.reg_weight = self.reg_schedule[iteration]

        return super().run_epoch(iteration)
