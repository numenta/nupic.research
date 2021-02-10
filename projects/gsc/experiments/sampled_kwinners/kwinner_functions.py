import torch
from torch.nn.functional import softmax
from torch.distributions.multinomial import Multinomial


def sampled_kwinners(x, k, temperature, relu=False, inplace=False):
    if k == 0:
        return torch.zeros_like(x)
    probs = softmax(x / temperature, dim=-1)
    dist = Multinomial(total_count=k, probs=probs)
    on_mask = dist.sample().bool()
    if relu:
        on_mask |= (x <= 0)
    off_mask = ~on_mask
    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)


def sampled_kwinners2d(x, k, temperature, relu=False, inplace=False):
    if k == 0:
        return torch.zeros_like(x)
    shape2 = (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    logits = x.view(shape2)
    probs = softmax(logits / temperature, dim=-1)
    dist = Multinomial(total_count=k, probs=probs)
    on_mask = dist.sample().bool()
    if relu:
        on_mask |= (logits <= 0)
    off_mask = ~on_mask
    off_mask = off_mask.view(x.shape)
    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)

__all__ = [
    "sampled_kwinners",
    "sampled_kwinners2d"
]