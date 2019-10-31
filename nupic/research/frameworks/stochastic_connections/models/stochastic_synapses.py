# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
from tqdm import tqdm

from nupic.research.frameworks.dynamic_sparse.models import BaseModel
from nupic.research.frameworks.stochastic_connections.binary_layers import (
    BinaryGatedConv2d,
    BinaryGatedLinear,
)
from nupic.research.frameworks.stochastic_connections.reparameterization_layers import (
    HardConcreteGatedConv2d,
    HardConcreteGatedLinear,
)

STOCHASTIC_MODULES = (
    BinaryGatedConv2d,
    BinaryGatedLinear,
    HardConcreteGatedConv2d,
    HardConcreteGatedLinear,
)


class StochasticSynapsesModel(BaseModel):

    def setup(self, config=None):
        super().setup(config)

        # Add specific defaults.
        new_defaults = dict(
            use_tqdm=False,
        )
        new_defaults = {k: v for k, v in new_defaults.items() if k not in self.__dict__}
        self.__dict__.update(new_defaults)

        # TODO: This could use a stricter condition
        # to ensure module is stochastic - not just having
        # a "l0_strength" perhaps by chance.
        self.named_stochastic_modules = [
            (name, module)
            for (name, module) in self.network.named_modules()
            if isinstance(module, STOCHASTIC_MODULES)
        ]

    @property
    def stochastic_modules(self):
        return [module for (_, module) in self.named_stochastic_modules]

    # -------------------
    # Procedural Methods
    # -------------------

    def _pre_epoch_setup(self):
        pass

    def _run_one_pass(self, loader, train=True, noise=False):

        # TODO: The logic in this function is mostly redundant to the logic in
        # `BaseModel._run_one_pass`. Could be good to merge the two.

        if self.use_tqdm:
            batches = tqdm(
                loader, leave=False, desc=("Training" if train else "Testing"))
        else:
            batches = loader

        epoch_loss = 0.
        correct = 0.

        for data, target in batches:
            data, target = data.to(self.device), target.to(self.device)
            output = self.network(data)
            loss = self.calc_loss(output, target, loader)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                epoch_loss += torch.sum(loss).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            for module in self.stochastic_modules:
                module.constrain_parameters()

        # Store loss and accuracy at each pass.
        loss = epoch_loss / len(loader.dataset)
        acc = correct / len(loader.dataset)
        self.logger.log_metrics(loss, acc, train, noise)

    def _post_epoch_updates(self, dataset=None):
        # TODO: Make custom logger to handle `nonzero_counts` functionality.
        # See 'dynamic_sparse/models/loggers.py'
        self.logger.log.update(self.nonzero_counts())

    def _post_optimize_updates(self):
        for module in self.stochastic_modules:
            module.constrain_parameters()

    # --------------------
    # Training Utils
    # --------------------

    def calc_loss(self, outputs, targets, loader):
        dataset_percent = loader.batch_size / len(loader.dataset)
        loss = self.loss_func(outputs, targets)
        loss += dataset_percent * self.regularization()
        return loss

    def regularization(self):
        reg = torch.tensor(0.).to(self.device)
        for module in self.stochastic_modules:
            reg += module.regularization()
        return reg

    def nonzero_counts(self):
        result = {}

        for name, module in self.named_stochastic_modules:

            num_inputs = 1.
            for d in module.weight_size()[1:]:
                num_inputs *= d

            e_nz_by_unit = module.get_expected_nonzeros()
            i_nz_by_unit = module.get_inference_nonzeros()

            multiplies, adds = module.count_inference_flops()

            result[name] = {
                "hist_expected_nz_by_unit": e_nz_by_unit.tolist(),
                "expected_nz": torch.sum(e_nz_by_unit).item(),
                "hist_inference_nz_by_unit": module.get_inference_nonzeros().tolist(),
                "inference_nz": torch.sum(i_nz_by_unit).item(),
                "num_input_units": num_inputs,
                "multiplies": multiplies,
                "flops": multiplies + adds,
            }

        return result
