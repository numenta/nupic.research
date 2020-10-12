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

import io

from nupic.research.frameworks.pytorch.model_utils import (
    deserialize_state_dict,
    serialize_state_dict,
)

__all__ = [
    "HasLRScheduler",
]


class HasLRScheduler:
    """
    Handles serialization of self.lr_scheduler. It does not create or step the
    LR scheduler.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.lr_scheduler = None

    def get_state(self):
        state = super().get_state()

        if self.lr_scheduler is not None:
            with io.BytesIO() as buffer:
                state_dict = self.lr_scheduler.state_dict()
                if "anneal_func" in state_dict:
                    # FIXME: This is a workaround for a PyTorch bug.
                    # https://github.com/pytorch/pytorch/issues/42376
                    del state_dict["anneal_func"]
                serialize_state_dict(buffer, state_dict)
                state["lr_scheduler"] = buffer.getvalue()

        return state

    def set_state(self, state):
        super().set_state(state)
        if "lr_scheduler" in state:
            with io.BytesIO(state["lr_scheduler"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            self.lr_scheduler.load_state_dict(state_dict)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        name = "HasLRScheduler"

        # Extended methods
        eo["setup_experiment"].append(name + ".setup_experiment")
        eo["get_state"].append(name + ": Get LR scheduler")
        eo["set_state"].append(name + ": Set LR scheduler")

        return eo
