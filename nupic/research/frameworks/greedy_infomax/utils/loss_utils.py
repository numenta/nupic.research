import torch
import torch.nn.functional as F


def multiple_cross_entropy(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    total_loss = torch.tensor(0.0, requires_grad=True)
    # Sum losses from each module
    for log_f_list, true_f_list in zip(log_f_module_list, true_f_module_list):
        # Sum losses for each k prediction
        for log_fk, true_fk in zip(log_f_list, true_f_list):
            total_loss = total_loss + F.cross_entropy(
                log_fk, true_fk, reduction=reduction
            )
    return total_loss


def module_specific_cross_entropy(data_lists, targets, reduction="mean", module=-1):
    log_f_module_list, true_f_module_list = data_lists
    total_loss = torch.tensor(0.0, requires_grad=True)
    # Sum losses from each module
    log_f_list, true_f_list = log_f_module_list[module], true_f_module_list[module]
    for log_fk, true_fk in zip(log_f_list, true_f_list):
        total_loss = total_loss + F.cross_entropy(log_fk, true_fk, reduction=reduction)
    return total_loss
