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
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nupic.research.frameworks.continual_learning.maml_utils import clone_model
from nupic.research.frameworks.pytorch.dataset_utils.samplers import (
    TaskDistributedSampler,
)
from nupic.research.frameworks.pytorch.model_utils import get_parent_module
from nupic.research.frameworks.vernon.distributed.experiments.components import (
    DistributedBase,
)
from nupic.research.frameworks.vernon.experiments.meta_cl_experiment import (
    MetaContinualLearningExperiment as MetaContinualLearningExperimentBase,
)

__all__ = [
    "MetaContinualLearningExperiment",
]


class MetaContinualLearningExperiment(DistributedBase,
                                      MetaContinualLearningExperimentBase):
    """
    ContinualLearningExperiment for distributed experiments. Distributed
    validation is not implemented.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        if self.distributed:
            self.model = DistributedDataParallel(self.model)
        else:
            self.model = DataParallel(self.model)

    def pre_epoch(self):
        super().pre_epoch()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)

    @classmethod
    def update_params(cls, named_params, model, loss, lr, distributed=False):
        """
        Takes a gradient step on the loss and updates the cloned parameters in place.
        """
        named_params = dict(named_params)
        params = list(named_params.values())
        gradients = torch.autograd.grad(
            loss, params,
            retain_graph=True, create_graph=True
        )

        if distributed:
            size = float(dist.get_world_size())
            for grad in gradients:
                dist.all_reduce(grad.data, op=dist.reduce_op.SUM)
                grad.data /= size

        if gradients is not None:
            for g, (name, p) in zip(gradients, named_params.items()):
                if g is not None:
                    updated = p.add(g, alpha=-lr)

                    # Update in-place in a way that preserves grads.
                    parent_module = get_parent_module(model, name)
                    base_name = name.split(".")[-1]
                    parent_module._parameters[base_name] = updated

    def adapt(self, cloned_adaptation_net, train_loss):
        named_fast_params = self.get_named_fast_params(cloned_adaptation_net)
        self.update_params(
            named_fast_params, cloned_adaptation_net, train_loss,
            self.adaptation_lr, distributed=self.distributed
        )

    @classmethod
    def create_sampler(cls, config, dataset, class_indices):
        """
        Provides a hook for a distributed experiment.
        """
        distributed = config.get("distributed", False)
        if distributed:
            return TaskDistributedSampler(
                dataset,
                class_indices
            )
        else:
            return super().create_sampler(config, dataset, class_indices)

    def clone_model(self, keep_as_reference=None):
        """
        Clones self.model by cloning some of the params and keeping those listed
        specified `keep_as_reference` via reference.
        """
        model = clone_model(self.model.module, keep_as_reference=None)

        if not self.distributed:
            model = DataParallel(model)
        else:
            # Instead of using DistributedDataParallel, the grads will be reduced
            # manually since we won't call loss.backward()
            model

        return model

    def get_model(self, clone=None):
        model = clone if clone is not None else self.model
        if hasattr(model, "module"):
            return model.module
        else:
            return model

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "DistributedMetaContinualLearningExperiment"

        # Overwritten methods
        eo["update_params"] = [exp + ".update_params"]
        eo["adapt"] = [exp + ".adapt"]
        eo["get_model"] = [exp + ".get_model"]
        eo["clone_model"] = [exp + ".clone_model"]

        # Extended methods
        eo["setup_experiment"].append(exp + ": DistributedDataParallel")
        eo["pre_epoch"].append(exp + ": Update distributed sampler")
        eo["create_sampler"].insert(0,
                                    ("if distributed { "
                                     "create distributed sampler"
                                     " } else { "))
        eo["create_sampler"].append("}")
        # FIXME: Validation is not currently distributed. Implement samplers in
        # a way that distributes validation, and implement aggregate_results.

        return eo
