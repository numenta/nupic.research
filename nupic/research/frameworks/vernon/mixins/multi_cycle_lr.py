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
            - multi_lr_scheduler_args: A list of (epoch, dict) pairs.
                                       The dicts don't need to include epoch
                                       counts, this is inferred from the config.
        """
        config = copy.deepcopy(config)

        ignored_class = config.pop("lr_scheduler_class", None)
        if ignored_class is not None and ignored_class != OneCycleLR:
            self.logger.warning("Ignoring lr_scheduler_class, using OneCycleLR")
        ignored_args = config.pop("lr_scheduler_args", None)
        if ignored_args is not None and len(ignored_args) > 0:
            self.logger.warning("Ignoring lr_scheduler_args, using "
                                "multi_lr_scheduler_args")

        super().setup_experiment(config)

        args_by_epoch = dict(copy.deepcopy(config["multi_lr_scheduler_args"]))

        # Insert epoch counts and div_factors
        cycle_start_epochs = sorted(args_by_epoch.keys())
        assert len(cycle_start_epochs) > 0 and cycle_start_epochs[0] == 0
        for i, start_epoch in enumerate(cycle_start_epochs):
            if i + 1 < len(cycle_start_epochs):
                end_epoch = cycle_start_epochs[i + 1]
            else:
                end_epoch = config["epochs"]

            args_by_epoch[start_epoch]["epochs"] = end_epoch - start_epoch

            # Default behavior: no sudden change in learning rate between
            # cycles.
            if "div_factor" not in args_by_epoch[start_epoch] and i > 0:
                prev_args = args_by_epoch[cycle_start_epochs[i - 1]]
                if "final_div_factor" in prev_args:
                    args_by_epoch[start_epoch]["div_factor"] = \
                        prev_args["final_div_factor"]
        self.multi_cycle_args_by_epoch = args_by_epoch

        if self.rank == 0:
            self.logger.info("MultiCycleLR regime: "
                             f"{self.multi_cycle_args_by_epoch}")

    def pre_epoch(self):
        super().pre_epoch()

        if self.current_epoch in self.multi_cycle_args_by_epoch:
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
