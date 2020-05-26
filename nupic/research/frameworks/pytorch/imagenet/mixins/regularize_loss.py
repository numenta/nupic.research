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


class RegularizeLoss(object):
    """
    Add a regularization term to the loss function.
    """
    def setup_experiment(self, config):
        """
        @param reg_schedule (list of tuples)
        Mapping from epoch number to the reg_weight to use on that timestep and
        afterward.

        @param downscale_reg_with_training_set (bool)
        If True, multiply the regularization term by (1 / size_of_training_set)
        """
        super().setup_experiment(config)

        if config.get("downscale_reg_with_training_set", False):
            self.reg_coefficient = (
                1 / len(self.train_loader.dataset)
            )
        else:
            self.reg_coefficient = 1

        reg_schedule = config.get("reg_schedule", None)
        if reg_schedule is None:
            self.reg_schedule = {}
            self.reg_weight = 1.0
        else:
            self.reg_schedule = dict(reg_schedule)
            self.reg_weight = self.reg_schedule[0]

        self._regularized_modules = [module
                                     for module in self.model.modules()
                                     if hasattr(module, "regularization")]

    def loss_function(self, *args, **kwargs):
        loss = super().loss_function(*args, **kwargs)

        if self.reg_weight != 0:
            # This is inner loop code, avoid any unnecessary tensor allocation.
            reg = None
            for module in self._regularized_modules:
                if reg is None:
                    reg = module.regularization()
                else:
                    reg += module.regularization()

            if reg is not None:
                reg *= self.reg_weight * self.reg_coefficient
                loss += reg

        return loss

    def pre_epoch(self):
        super().pre_epoch()
        if self.current_epoch in self.reg_schedule:
            self.reg_weight = self.reg_schedule[self.current_epoch]

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("RegularizeLoss initialization")
        eo["loss_function"].append("RegularizeLoss")
        eo["pre_epoch"].append("RegularizeLoss update regularization weight")
        return eo
