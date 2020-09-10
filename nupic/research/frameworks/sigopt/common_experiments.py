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

from .sigopt_experiment import SigOptExperiment


class SigOptImagenetExperiment(SigOptExperiment):
    """
    A subclass of SigOptExperiment used to sit between an experiment runner (such as
    Ray) and the ImagenetExperiment class. update_config_with_suggestion() is specific
    to our ImagenetExperiment config.
    """

    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update our various config dicts properly so that we
        can pass it onto ImagenetExperiment.
        """
        assignments = suggestion.assignments

        assert "optimizer_args" in config
        assert "lr_scheduler_args" in config

        # For multi-task experiments where epoch is the task. Must have a metadata
        # field called max_epochs.
        if suggestion.task is not None and "epoch" in suggestion.task.name:
            max_epochs = self.sigopt_config["metadata"]["max_epochs"]
            epochs = int(max_epochs * suggestion.task.cost)
            print("Suggested task/cost/epochs for this multitask experiment: ",
                  suggestion.task.name, suggestion.task.cost, epochs)
            config["epochs"] = epochs
            config["lr_scheduler_args"]["epochs"] = epochs

        # Optimizer args
        if "log_lr" in assignments:
            config["optimizer_args"]["lr"] = math.exp(assignments["log_lr"])
            assignments.pop("log_lr")

        if "momentum" in assignments:
            config["optimizer_args"]["momentum"] = assignments["momentum"]
            config["lr_scheduler_args"]["max_momentum"] = assignments["momentum"]
            assignments.pop("momentum")

        if "weight_decay" in assignments:
            config["optimizer_args"]["weight_decay"] = assignments["weight_decay"]
            assignments.pop("weight_decay")

        # lr_scheduler args
        if "gamma" in assignments:
            config["lr_scheduler_args"]["gamma"] = assignments["gamma"]
            assignments.pop("gamma")

        if "step_size" in assignments:
            config["lr_scheduler_args"]["step_size"] = assignments["step_size"]
            assignments.pop("step_size")

        # Parameters for OneCycleLR
        if "pct_start" in assignments:
            # Ensure integer number of epochs in pct start to avoid crash.
            start_epochs = int(assignments["pct_start"] * config["epochs"] + 0.5)
            pct_start = float(start_epochs) / config["epochs"]
            config["lr_scheduler_args"]["pct_start"] = pct_start
            assignments.pop("pct_start")

        if "cycle_momentum" in assignments:
            config["lr_scheduler_args"]["cycle_momentum"] = \
                eval(assignments["cycle_momentum"])
            assignments.pop("cycle_momentum")

        if "max_lr" in assignments or "init_lr" in assignments:
            # From the OneCycleLR docs:
            #   initial_lr = max_lr/div_factor
            #   min_lr = initial_lr/final_div_factor
            max_lr = assignments.get("max_lr", 6.0)
            init_lr = assignments.get("init_lr", 1.0)

            config["lr_scheduler_args"]["max_lr"] = max_lr
            config["lr_scheduler_args"]["div_factor"] = max_lr / init_lr
            config["lr_scheduler_args"]["final_div_factor"] = init_lr / 0.00025
            assignments.pop("init_lr", None)
            assignments.pop("max_lr", None)

        config.update(assignments)
