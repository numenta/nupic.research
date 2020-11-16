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

from nupic.research.frameworks.pytorch.restore_utils import load_state_from_checkpoint

__all__ = [
    "DelayLoadCheckpoint",
]


class DelayLoadCheckpoint:
    """
    Load the checkpoint after super().create_model has finished.
    """
    @classmethod
    def create_model(cls, config, device):
        model = super().create_model({**config, "checkpoint_file": None},
                                     device)
        load_state_from_checkpoint(model, config.get("checkpoint_file", None),
                                   device)
        return model

    @classmethod
    def get_execution_order(cls):
        name = "DelayLoadCheckpoint"
        eo = super().get_execution_order()
        eo["create_model"].insert(0, name + ": Set checkpoint_file=None")
        eo["create_model"].append(0, name + ": Load checkpoint_file")
