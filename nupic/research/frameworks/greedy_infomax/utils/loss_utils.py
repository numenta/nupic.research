import torch
import torch.nn.functional as F


def multiple_cross_entropy(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    for log_f_list, true_f_list in zip(log_f_module_list, true_f_module_list):
        # Sum losses for each k prediction
        for log_fk, true_fk in zip(log_f_list, true_f_list):
            total_loss = total_loss + F.cross_entropy(
                log_fk, true_fk, reduction=reduction
            )
    return total_loss

def multiple_log_softmax_nll_loss(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    for log_f_list, true_f_list in zip(log_f_module_list, true_f_module_list):
        # Sum losses for each k prediction
        for log_fk, true_fk in zip(log_f_list, true_f_list):
            softmax_fk = torch.softmax(log_fk, dim=1)
            log_softmax_fk = torch.log(softmax_fk + 1e-11)
            total_loss = total_loss + F.nll_loss(
                log_softmax_fk, true_fk, reduction=reduction
            )
    return total_loss

def true_GIM_loss(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    for log_f_list, true_f_list in zip(log_f_module_list, true_f_module_list):
        # Sum losses for each k prediction
        for log_fk, true_fk in zip(log_f_list, true_f_list):
            numerator = log_fk[:, 0, :, :]
            denominator = torch.logsumexp(log_fk[:, 1:, :, :], dim=1).mean()
            total_loss = total_loss + (numerator-denominator).mean()
    return total_loss


def module_specific_cross_entropy(data_lists, targets, reduction="mean", module=-1):
    log_f_module_list, true_f_module_list = data_lists
    device = data_lists[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    log_f_list, true_f_list = log_f_module_list[module], true_f_module_list[module]
    for log_fk, true_fk in zip(log_f_list, true_f_list):
        total_loss = total_loss + F.cross_entropy(log_fk, true_fk, reduction=reduction)
    return total_loss

