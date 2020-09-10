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

from torch.optim.lr_scheduler import StepLR


class StepLRParams(object):

    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update the lr_scheduler_args with StepLR params.

        :param config:
            - lr_scheduler_args: dict of lr arguments
            - optimizer_args: dict of optimizer arguments
        :param suggestion:
            - assignments (all optional)
                - gamma
                - step_size
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments

        assert "lr_scheduler_args" in config
        assert "lr_scheduler_class" in config

        lr_scheduler_args = config["lr_scheduler_args"]
        lr_scheduler_class = config["lr_scheduler_class"]

        assert lr_scheduler_class == StepLR

        # lr_scheduler args
        if "gamma" in assignments:
            lr_scheduler_args["gamma"] = assignments["gamma"]

        if "step_size" in assignments:
            lr_scheduler_args["step_size"] = assignments["step_size"]
