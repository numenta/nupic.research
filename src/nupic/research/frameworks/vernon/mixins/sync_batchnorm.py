# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
import torch.nn as nn


class SyncBatchNorm:
    """
    This mixin converts the BatchNorm modules to SyncBatchNorm modules when utilizing
    distributed training on GPUs.

    Example config:
        config=dict(
            use_sync_batchnorm=True
        )
    """
    def create_model(self, config, device):
        model = super().create_model(config, device)
        use_sync_batchnorm = config.get("use_sync_batchnorm", True)
        distributed = config.get("distributed", False)
        if use_sync_batchnorm and distributed and next(model.parameters()).is_cuda:
            # Convert batch norm to sync batch norms
            model = nn.modules.SyncBatchNorm.convert_sync_batchnorm(module=model)
        return model

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].insert(0, "Sync Batchnorm begin")
        eo["setup_experiment"].append("Sync Batchnorm end")
        return eo
