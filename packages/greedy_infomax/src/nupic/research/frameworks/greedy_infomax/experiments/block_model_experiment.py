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
import copy
import itertools

import torch

from nupic.research.frameworks.greedy_infomax.models.block_model import BlockModel
from nupic.research.frameworks.greedy_infomax.models.utility_layers import EmitEncoding
from nupic.research.frameworks.greedy_infomax.utils.train_utils import (
    aggregate_eval_results_gim,
    evaluate_gim_model,
    train_gim_model,
)
from nupic.research.frameworks.self_supervised_learning.experiments import (
    SelfSupervisedExperiment,
)
from nupic.research.frameworks.vernon import mixins
from nupic.research.frameworks.vernon.network_utils import create_model


class BlockModelExperiment(
    mixins.LogEveryLoss,
    mixins.LogEveryLearningRate,
    mixins.LogBackpropStructure,
    mixins.RezeroWeights,
    SelfSupervisedExperiment,
):
    """
    NOTE: This experiment class is deprecated.

    The BlockModel experiment class is used in the GreedyInfoMax context. It only
    supports the use of the BlockModel which is essentially a more sophisticated
    version of nn.Sequential. Modules are specified in the config and can be
    loaded independently, but this only supports models which follow a strictly
    sequential computation graph. It was for this reason that the BlockModel and the
    BlockModelExperiment have been deprecated in favor of the more general
    GreedyInfoMax experiment.

    Because logging is generally always included in Greedy InfoMax experiments,
    this experiment class inherits from standard logging mixins and extends their
    functionality to log the losses and accuracies of multiple modules.

    The BlockModel is entirely defined in the config, as opposed to a standard model
    whose structure is designed in a single class. This gives the BlockModel a
    certain amount of flexibility as modules are saved and loaded independently. An
    example config might look like:

    ```
    config = dict(
        model_class="BlockModel",
        model_args=dict(
            module_args=[
                dict(
                    module_class="nn.Linear",
                    module_args=dict(
                        in_features=784,
                        out_features=128,
                    ),
                ),
                dict(
                    module_class="nn.ReLU",
                ),
                dict(
                    module_class="BilinearInfo",
                    module_args=dict( *BilinearInfoArgs*)
                ),
            ]
        ),

    This BlockModel would have the following structure:
    nn.Linear(784, 128)
    nn.ReLU()
    BilinearInfo()

    This can get to be very complicated for things like large ResNets, so the default
    set of ResNets with various sparsity levels have been pre-defined in
    greedy_infomax/utils/model_utils.py.
    """
    def setup_experiment(self, config):
        emit_encoding_channels = [
            x["model_args"]["channels"]
            for x in config["model_args"]["module_args"]
            if x["model_class"] == EmitEncoding
        ]
        config["classifier_config"]["model_args"].update(
            in_channels=emit_encoding_channels
        )
        super().setup_experiment(config)
        self.evaluate_model = evaluate_gim_model
        self.train_model = self.train_model_supervised = train_gim_model
        self.multiple_module_loss_history = []

    @classmethod
    def create_model(cls, config, device):
        """
        Creates the BlockModel. The config should specify the model class as
        BlockModel, and then the model_args should contain a "module_args" parameter
        which contains a list of dictionaries that specify submodules. A simple
        example might look like this:

        """
        if config["model_class"] != BlockModel:
            return super().create_model(config, device)
        model_args = config.get("model_args", {})
        module_args = model_args.get("module_args", [])
        modules = []
        for module_dict in module_args:
            modules.append(
                create_model(
                    model_class=module_dict["model_class"],
                    model_args=module_dict.get("model_args", {}),
                    init_batch_norm=module_dict.get("init_batch_norm", False),
                    device=device,
                    checkpoint_file=module_dict.get("checkpoint_file", None),
                    load_checkpoint_args=module_dict.get("load_checkpoint_args", {}),
                )
            )
        model_args["modules"] = modules
        return create_model(
            model_class=config["model_class"],
            model_args=model_args,
            init_batch_norm=config.get("init_batch_norm", False),
            device=device,
            checkpoint_file=config.get("checkpoint_file", None),
            load_checkpoint_args=config.get("load_checkpoint_args", {}),
        )

    @classmethod
    def create_optimizer(cls, config, device):
        if config["model_class"] != BlockModel:
            return super().create_optimizer(config, device)
        model_args = config.get("model_args", {})
        module_args = model_args.get("module_args", [])
        module_instances = model_args["modules"]
        parameters_to_train = []
        for module_dict, module_instance in zip(module_args, module_instances):
            if module_dict.get("train", True):
                parameters_to_train.append(module_instance.parameters())
        parameters_to_train = itertools.chain(*parameters_to_train)
        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        return optimizer_class(parameters_to_train, **optimizer_args)

    def post_batch(self, error_loss, complexity_loss, batch_idx, **kwargs):
        super().post_batch(
            error_loss=error_loss,
            complexity_loss=complexity_loss,
            batch_idx=batch_idx,
            **kwargs,
        )
        if self.should_log_batch(batch_idx) and "module_losses" in kwargs.keys():
            self.multiple_module_loss_history.append(kwargs["module_losses"].clone())

    def run_epoch(self):
        result = super().run_epoch()
        if len(self.multiple_module_loss_history) > 0:
            log = torch.stack(self.multiple_module_loss_history)
            module_loss_history = log.cpu().numpy()
            for i in range(log.shape[1]):
                result[f"module_{i}_loss_history"] = module_loss_history[:, i].tolist()
            self.multiple_module_loss_history = []
            result["num_bilinear_info_modules"] = int(log.shape[1])
        return result

    @classmethod
    def get_readable_result(cls, result):
        return result

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        result_by_timestep = super().expand_result_to_time_series(result, config)
        recorded_timesteps = cls.get_recorded_timesteps(result, config)
        for i in range(result["num_bilinear_info_modules"]):
            for t, loss in zip(recorded_timesteps, result[f"module_{i}_loss_history"]):
                result_by_timestep[t].update({f"module_{i}_train_loss": loss})
        return result_by_timestep

    @classmethod
    def _aggregate_validation_results(cls, results):
        result = copy.copy(results[0])
        result.update(aggregate_eval_results_gim(results))
        return result

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("BlockModelExperiment: initialize")
        eo["create_optimizer"].append("BlockModelExperiment: create optimizer")
        eo["post_batch"].append("BlockModelExperiment: record losses")
        eo["run_epoch"].append("BlockModelExperiment: to result dict")
        eo["expand_result_to_time_series"].append("BlockModelExperiment: module_losses")
        return eo
