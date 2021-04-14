import torch.nn as nn
import torch

def multiple_cross_entropy(data_lists, targets):
    y_list, target_list = data_lists
    losses = [torch.nn.functional.cross_entropy(y, target) for y, target in zip(
        y_list, target_list)]
    return torch.sum(losses)