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

import math

from torch.optim import SGD


class SGDParams(object):

    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update the optimizer_args with SGD optimizer params.

        :param config:
            - optimizer_args: dict of optimizer arguments
            - optimizer_class: SGD optimizer
        :param suggestion:
            - assignments (all optional)
                - log_lr
                - momentum
                - weight_decay
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments

        assert "optimizer_args" in config
        assert "optimizer_class" in config

        optimizer_args = config["optimizer_args"]
        optimizer_class = config["optimizer_class"]

        assert optimizer_class == SGD

        # Optimizer args
        if "log_lr" in assignments:
            optimizer_args["lr"] = math.exp(assignments["log_lr"])

        if "momentum" in assignments:
            optimizer_args["momentum"] = assignments["momentum"]

        if "weight_decay" in assignments:
            optimizer_args["weight_decay"] = assignments["weight_decay"]
