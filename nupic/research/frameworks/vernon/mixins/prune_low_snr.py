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

import torch


class PruneLowSNRLayers:
    """
    Prunes weights layer-wise by signal-to-noise ratio. Requires variational dropout
    modules. Annotate each module with its their target density by setting
    attribute module._target_density.
    """

    def setup_experiment(self, config):
        """
        :param config:
            - prune_schedule: A list of (epoch, progress) pairs, where progress
                              denotes how far toward the target sparsity.
                              The progress is typically 1.0 at the end.
            - prune_curve_shape: Describes how to interpret the "progess" in
                                 prune_schedule. With "exponential", increasing
                                 progress at a fixed rate causes a fixed
                                 percentage if remaining weights to be pruned at
                                 each step. With "linear", a fixed percentage of
                                 the total weights will be pruned at each step.
            - validate_on_prune: Whether to run validation after pruning.
        """
        super().setup_experiment(config)
        self.prune_schedule = dict(config["prune_schedule"])
        self.prune_curve_shape = config.get("prune_curve_shape", "exponential")
        self.validate_on_prune = config.get("validate_on_prune", False)

    def pre_epoch(self):
        super().pre_epoch()
        if self.current_epoch in self.prune_schedule:
            prune_progress = self.prune_schedule[self.current_epoch]
            model = self.model
            if hasattr(model, "module"):
                model = model.module
            vdrop_data = model.vdrop_central_data

            for (module, z_mask, z_mu, z_logvar, z_logalpha) in zip(
                vdrop_data.modules,
                vdrop_data.z_mask.split(vdrop_data.z_chunk_sizes),
                vdrop_data.z_mu.split(vdrop_data.z_chunk_sizes),
                vdrop_data.z_logvar.split(vdrop_data.z_chunk_sizes),
                vdrop_data.compute_z_logalpha().split(vdrop_data.z_chunk_sizes),
            ):

                if hasattr(module, "_target_density"):
                    if self.prune_curve_shape == "exponential":
                        density = module._target_density ** prune_progress
                    elif self.prune_curve_shape == "linear":
                        density = 1 - ((1 - module._target_density) * prune_progress)

                    num_weights = math.floor(z_logalpha.numel() * density)
                    on_indices = z_logalpha.topk(num_weights, largest=False)[1]

                    z_mask.zero_()
                    z_mask[on_indices] = 1
                    z_mu.data *= z_mask
                    z_logvar[~z_mask.bool()] = vdrop_data.pruned_logvar_sentinel

                    if not self.logger.disabled:
                        name = [
                            name
                            for name, m in self.model.named_modules()
                            if m is module
                        ][0]
                        self.logger.info(f"Pruned {name} to {density} ")

            for parameter, mask in vdrop_data.masked_parameters():
                zero_momentum(self.optimizer, parameter, mask)

            self.current_timestep += 1

            if self.validate_on_prune:
                result = self.validate()
                self.extra_val_results.append((self.current_timestep, result))

    def run_epoch(self):
        results = super().run_epoch()
        vdrop_data = self.model.module.vdrop_data
        total_prunable_parameters = vdrop_data.z_mask.size(0)
        remaining_nonzero_parameters = vdrop_data.z_mask.double().sum().item()
        model_density = remaining_nonzero_parameters / total_prunable_parameters
        results["total_prunable_parameters"] = total_prunable_parameters
        results["remaining_nonzero_parameters"] = remaining_nonzero_parameters
        results["model_density"] = model_density
        return results

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("PruneLowSNRLayers: Initialize")
        eo["pre_epoch"].append("PruneLowSNRLayers: Maybe prune")
        eo["run_epoch"].append("PruneLowSNRLayers: Log parameter sparsity")
        return eo


class PruneLowSNRGlobal:
    """
    Prunes weights by signal-to-noise ratio across the entire model. Requires
    variational dropout modules. For this mixin to work, the model must have a
    VDropCentralData module stored in the model as self.vdrop_central_data.
    """

    def setup_experiment(self, config):
        """
        :param config:
            - prune_schedule: A list of (epoch, density) pairs.
            - validate_on_prune: Whether to run validation after pruning.
        """
        super().setup_experiment(config)
        self.prune_schedule = dict(config["prune_schedule"])
        self.validate_on_prune = config.get("validate_on_prune", False)
        self.log_module_sparsities = config.get("log_module_sparsities", False)
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        assert all(
            num_weights == 1 for num_weights in model.vdrop_central_data.z_num_weights
        ), "This mixin does not support group sparsity"

    def pre_epoch(self):
        super().pre_epoch()
        if self.current_epoch in self.prune_schedule:
            target_density = self.prune_schedule[self.current_epoch]
            model = self.model
            if hasattr(model, "module"):
                model = model.module
            vdrop_data = self.model.vdrop_central_data

            z_logalpha = vdrop_data.compute_z_logalpha()
            z_mask = vdrop_data.z_mask
            z_mu = vdrop_data.z_mu
            z_logvar = vdrop_data.z_logvar

            num_weights = math.floor(z_logalpha.numel() * target_density)
            on_indices = z_logalpha.topk(num_weights, largest=False)[1]

            z_mask.zero_()
            z_mask[on_indices] = 1
            z_mu.data *= z_mask
            with torch.no_grad():
                z_logvar[~z_mask.bool()] = vdrop_data.pruned_logvar_sentinel

            self.logger.info(f"Pruned global model to {target_density} ")

            for parameter, mask in vdrop_data.masked_parameters():
                zero_momentum(self.optimizer, parameter, mask)

            self.current_timestep += 1

            if self.validate_on_prune:
                result = self.validate()
                self.extra_val_results.append((self.current_timestep, result))

    def run_epoch(self):
        results = super().run_epoch()
        vdrop_data = self.model.vdrop_central_data
        total_prunable_parameters = vdrop_data.z_mask.size(0)
        remaining_nonzero_parameters = vdrop_data.z_mask.double().sum().item()
        model_density = remaining_nonzero_parameters / total_prunable_parameters
        results["total_prunable_parameters"] = total_prunable_parameters
        results["remaining_nonzero_parameters"] = remaining_nonzero_parameters
        results["model_density"] = model_density
        if self.log_module_sparsities:
            module_idx = 0
            for module, z_mask in zip(
                vdrop_data.modules, vdrop_data.z_mask.split(vdrop_data.z_chunk_sizes)
            ):
                density = z_mask.sum() / z_mask.numel()
                module_class = module.__class__.__name__
                results[
                    str(module_class) + "_" + str(module_idx) + "_density"
                ] = density
                module_idx += 1
        return results

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("PruneLowSNRGlobal: Initialize")
        eo["pre_epoch"].append("PruneLowSNRGlobal: Maybe prune")
        eo["run_epoch"].append("PruneLowSNRGlobal: Log parameter sparsity")
        return eo


def zero_momentum(optimizer, parameter, mask):
    """
    Modifies the momentum of the given parameter by masking it.
    """
    if isinstance(optimizer, torch.optim.Adam):
        state = optimizer.state[parameter]
        if "exp_avg" in state:
            state["exp_avg"].data *= mask
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].data *= mask
    elif isinstance(optimizer, torch.optim.SGD):
        state = optimizer.state[parameter]
        if state.get("momentum_buffer") is not None:
            state["momentum_buffer"] *= mask
    else:
        raise ValueError(
            f"Tell me how to zero the momentum of optimizer {type(optimizer)}"
        )
