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

try:
    from apex import amp
except ImportError:
    amp = None


__all__ = [
    "MixedPrecision",
]


class MixedPrecision:
    """
    Adds support for apex mixed precision.
    """
    def setup_experiment(self, config):
        """
        :param config: Dictionary containing the configuration parameters

            - mixed_precision: Whether or not to enable apex mixed precision
            - mixed_precision_args: apex mixed precision arguments.
                                    See "amp.initialize"
        """
        super().setup_experiment(config)

        # Validate mixed precision requirements
        self.mixed_precision = config.get("mixed_precision", False)
        if self.mixed_precision and amp is None:
            self.mixed_precision = False
            self.logger.error(
                "Mixed precision requires NVIDIA APEX."
                "Please install apex from https://www.github.com/nvidia/apex"
                "Disabling mixed precision training.")

        # Configure mixed precision training
        if self.mixed_precision:
            amp_args = config.get("mixed_precision_args", {})
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, **amp_args)
            self.logger.info("Using mixed precision")

    def get_state(self):
        state = super().get_state()

        if self.mixed_precision:
            with io.BytesIO() as buffer:
                serialize_state_dict(buffer, amp.state_dict())
                state["amp"] = buffer.getvalue()

        return state

    def set_state(self, state):
        super().set_state(state)

        if "amp" in state and amp is not None:
            with io.BytesIO(state["amp"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            amp.load_state_dict(state_dict)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        name = "MixedPrecision"

        # Extended methods
        eo["setup_experiment"].append(name + ".setup_experiment")
        eo["get_state"].append(name + ": Get amp")
        eo["set_state"].append(name + ": Set amp")

        return eo
