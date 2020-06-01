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

from torch.nn.modules.batchnorm import _BatchNorm


class CustomBatchNorm(object):
    """
    Allows customization of batch norm parameters
    """
    def setup_experiment(self, config):
        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters
            - batch_norm_momentum: Momentum for the batch norm layers
        """
        super().setup_experiment(config)

        # set batch norm momentum equally to all layers
        batch_norm_momentum = config.get("batch_norm_momentum", 0.1)
        self.logger.info(f"Setting batch norm momentum to {batch_norm_momentum}")
        for m in self.model.modules():
            if isinstance(m, _BatchNorm):
                m.momentum = batch_norm_momentum

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append(
            "Customization of batch norm layers")
        return eo
