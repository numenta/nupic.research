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

import torch


class LogCovariance(object):
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.log_covariance_layernames = config.get("log_covariance_layernames",
                                                    ())

    def validate(self, *args, **kwargs):
        activations = {layername: []
                       for layername in self.log_covariance_layernames}

        def accumulator(layername):
            def accumulate_activation(module, x, y):
                activations[layername].append(y)

            return accumulate_activation

        hooks = [getattr(self.module, layername)
                 .register_forward_hook(accumulator(layername))
                 for layername in self.log_covariance_layernames]
        result = super().test(*args, **kwargs)
        for hook in hooks:
            hook.remove()

        for layername, layer_activations in activations.items():
            H = torch.cat(layer_activations) # NOQA N806
            H -= torch.mean(H, dim=0)
            cov = H.t().mm(H) / H.shape[0]
            var = cov.diag()
            # Mask out the diagonal
            cov *= (1 - torch.eye(cov.shape[0]).to(self.device))
            result["{}/covariance_sum_of_squares".format(layername)] = \
                (cov.pow(2).sum() / 2).item()
            result["{}/variance_sum".format(layername)] = var.sum().item()

        return result
