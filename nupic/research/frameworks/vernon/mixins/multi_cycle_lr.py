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

import copy

from torch.optim.lr_scheduler import OneCycleLR

from nupic.research.frameworks.vernon.experiment_utils import create_lr_scheduler

class MultiCycleLR:
    """
    Composes a sequence of OneCycleLR regimes, allowing different configurations
    for each cycle. This infers args like total_batches, epochs, and also the
    div_factor for subsequent cycles.
    """
    def setup_experiment(self, config):
        """
        :param config:
            - multi_cycle_lr_args: A list of (epoch, dict) pairs.
                                   The dicts don't need to include epoch
                                   counts, this is inferred from the config.
        """
        config = copy.deepcopy(config)

        ignored_class = config.pop("lr_scheduler_class", None)
        ignored_args = config.pop("lr_scheduler_args", None)

        config["lr_scheduler_step_every_batch"] = True

        super().setup_experiment(config)

        if ignored_class is not None and ignored_class != OneCycleLR:
            self.logger.warning("Ignoring lr_scheduler_class, using OneCycleLR")
        if ignored_args is not None and len(ignored_args) > 0:
            self.logger.warning("Ignoring lr_scheduler_args, using "
                                "multi_cycle_lr_args")

        # Insert epoch counts and div_factors
        improved_args = {}
        multi_cycle_lr_args = sorted(config["multi_cycle_lr_args"],
                                     key=lambda x: x[0])
        for i, (start_epoch, cycle_config) in enumerate(multi_cycle_lr_args):
            if i + 1 < len(multi_cycle_lr_args):
                end_epoch = multi_cycle_lr_args[i + 1][0]
            else:
                end_epoch = config["epochs"]

            cycle_config = copy.deepcopy(cycle_config)
            cycle_config["epochs"] = end_epoch - start_epoch

            # Default behavior: no sudden change in learning rate between
            # cycles.
            if "div_factor" not in cycle_config and i > 0:
                prev_cycle_config = multi_cycle_lr_args[i - 1][1]
                if "final_div_factor" in prev_cycle_config:
                    cycle_config["div_factor"] = \
                        prev_cycle_config["final_div_factor"]

            improved_args[start_epoch] = cycle_config

        self.multi_cycle_args_by_epoch = improved_args

        self.logger.info("MultiCycleLR regime: "
                         f"{self.multi_cycle_args_by_epoch}")

        # Set it immediately, rather than waiting for the pre_epoch, in case a
        # restore is occurring.
        args = self.multi_cycle_args_by_epoch[0]
        self.lr_scheduler = create_lr_scheduler(
            optimizer=self.optimizer,
            lr_scheduler_class=OneCycleLR,
            lr_scheduler_args=args,
            steps_per_epoch=self.total_batches)

    def pre_epoch(self):
        super().pre_epoch()

        if self.current_epoch != 0 and \
           self.current_epoch in self.multi_cycle_args_by_epoch:

            args = self.multi_cycle_args_by_epoch[self.current_epoch]
            self.lr_scheduler = create_lr_scheduler(
                optimizer=self.optimizer,
                lr_scheduler_class=OneCycleLR,
                lr_scheduler_args=args,
                steps_per_epoch=self.total_batches)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].insert(
            0, "MultiCycleLR: Prevent LR scheduler from being constructed")
        eo["setup_experiment"].append("MultiCycleLR: Initialize")
        eo["pre_epoch"].append("MultiCycleLR: Maybe initialize lr_scheduler")
        return eo
