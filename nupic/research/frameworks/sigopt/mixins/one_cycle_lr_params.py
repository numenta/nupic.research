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

from torch.optim.lr_scheduler import OneCycleLR


class OneCycleLRParams(object):

    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update the lr_scheduler_args with OneCycleLR params.

        :param config:
            - lr_scheduler_args: dict of lr arguments
            - lr_scheduler_class: OneCycleLR
        :param suggestion:
            - assignments (all optional)
                - momentum: identifies max_momentum
                - pct_start: transformed to (pct_start/epochs + 0.5) / epochs
                - cycle_momentum
                - final_div_factor
                - max_lr
                - div_factor
                - init_lr: sets div_factor = max_lr / init_lr, if it's not given
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments

        assert "lr_scheduler_args" in config
        assert "lr_scheduler_class" in config

        lr_scheduler_args = config["lr_scheduler_args"]
        lr_scheduler_class = config["lr_scheduler_class"]

        assert lr_scheduler_class == OneCycleLR

        # For multi-task experiments where epoch is the task. Must have a metadata
        # field called max_epochs.
        if suggestion.task is not None and "epoch" in suggestion.task.name:
            max_epochs = self.sigopt_config["metadata"]["max_epochs"]
            epochs = int(max_epochs * suggestion.task.cost)
            lr_scheduler_args["epochs"] = epochs

        if "momentum" in assignments:
            lr_scheduler_args["max_momentum"] = assignments["momentum"]

        if "pct_start" in assignments:
            # Ensure integer number of epochs in pct start to avoid crash.
            start_epochs = int(assignments["pct_start"] * config["epochs"] + 0.5)
            pct_start = float(start_epochs) / config["epochs"]
            lr_scheduler_args["pct_start"] = pct_start

        if "cycle_momentum" in assignments:
            lr_scheduler_args["cycle_momentum"] = eval(assignments["cycle_momentum"])

        if "div_factor" in assignments:
            assert "init_lr" not in assignments, (
                "Only one of init_lr of div_factor should be given"
            )
            lr_scheduler_args["div_factor"] = assignments["div_factor"]

        if "final_div_factor" in assignments:
            lr_scheduler_args["final_div_factor"] = assignments["assignments"]

        if "max_lr" in assignments:
            max_lr = assignments["max_lr"]
            lr_scheduler_args["max_lr"] = max_lr

            if "init_lr" in assignments:
                # initial_lr = max_lr / div_factor
                init_lr = assignments["init_lr"]
                lr_scheduler_args["div_factor"] = max_lr / init_lr

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["update_config_with_suggestion"].append(
            "OneCycleLRParams.update_config_with_suggestion"
        )
        return eo
