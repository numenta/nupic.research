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
#
# This work was based on the original Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------

import torch
import torch.nn.functional as F


"""
Used during unsupervised training of any GIM model. Returns the sum of all 
BilinearInfo cross entropy losses.
"""
def multiple_cross_entropy_bilinear_info(log_f_module_list, targets, reduction="mean"):
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    for log_f_list in log_f_module_list:
        # Sum losses for each k prediction
        for log_fk in log_f_list:
            # Positive samples are at index 0
            true_fk = torch.zeros(
                (log_fk.shape[0], log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=log_fk.device,
                requires_grad=False,
            )  # b, y, x
            total_loss = total_loss + F.cross_entropy(
                log_fk, true_fk, reduction=reduction
            )
    return total_loss

"""
Used when training BlockModels using GIM. Returns a tensor of losses, each entry 
representing the cross entropy loss of a specific BilinearInfo module.
"""
def all_module_multiple_cross_entropy_bilinear_info(log_f_module_list, targets,
                                           reduction="mean"):
    device = log_f_module_list[0][0].device
    module_losses = torch.zeros(len(log_f_module_list),
                                requires_grad=True,
                                device=device)
    # Sum losses from each module
    for i, log_f_list in enumerate(log_f_module_list):
        # Sum losses for each k prediction
        for log_fk in log_f_list:
            # Positive samples are at index 0
            true_fk = torch.zeros(
                (log_fk.shape[0], log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=log_fk.device,
                requires_grad=False,
            ) # b, y, x
            module_losses.index_add(0,
                                    torch.tensor([i], dtype=torch.long,
                                                 device=log_fk.device),
                                    F.cross_entropy(log_fk, true_fk,
                                                    reduction=reduction)
                                    )
    return module_losses


"""
Used for supervised training of a BlockModel with GIM. This outputs a tensor of losses,
each of which is the cross entropy classification loss according to a specific 
EmitEncoding module paired with a classification head.
"""
def multiple_cross_entropy(outputs, targets, reduction="sum"):
    device = outputs.device
    module_losses = torch.zeros(outputs.shape[0], requires_grad=True, device=device)
    for i in range(outputs.shape[0]):
        module_losses[i] += F.cross_entropy(outputs[i], targets)
    return module_losses




def multiple_log_softmax_nll_loss(data_lists, targets, reduction="mean"):
    return module_specific_log_softmax_nll_loss(data_lists, targets).sum()


def module_specific_log_softmax_nll_loss(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.zeros(len(log_f_module_list), requires_grad=False, device=device)
    # Sum losses from each module
    for i, (log_f_list, true_f_list) in enumerate(
        zip(log_f_module_list, true_f_module_list)
    ):
        # Sum losses for each k prediction
        for log_fk, true_fk in zip(log_f_list, true_f_list):
            softmax_fk = torch.softmax(log_fk, dim=1)
            log_softmax_fk = torch.log(softmax_fk + 1e-11)
            total_loss[i] = total_loss[i] + F.nll_loss(
                log_softmax_fk, true_fk, reduction=reduction
            )
        total_loss[i] /= len(log_f_list)

    return total_loss


def true_gim_loss(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    for log_f_list, true_f_list in zip(log_f_module_list, true_f_module_list):
        # Sum losses for each k prediction
        for log_fk, _ in zip(log_f_list, true_f_list):
            numerator = log_fk[:, 0, :, :]
            denominator = torch.logsumexp(log_fk[:, 1:, :, :], dim=1).mean()
            total_loss = total_loss + (numerator - denominator).mean()
    return total_loss


def module_specific_cross_entropy(data_lists, targets, reduction="mean", module=-1):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    log_f_list, true_f_list = log_f_module_list[module], true_f_module_list[module]
    for log_fk, true_fk in zip(log_f_list, true_f_list):
        total_loss = total_loss + F.cross_entropy(log_fk, true_fk, reduction=reduction)
    return total_loss
