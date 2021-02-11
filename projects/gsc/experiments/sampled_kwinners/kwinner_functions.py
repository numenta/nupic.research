import torch
from torch.nn.functional import softmax
from torch.distributions.multinomial import Multinomial


def sampled_kwinners(x, k, temperature, relu=False, inplace=False):
    """
    A stochastic K-winner take all function for creating layers with sparse output.
    Keeps only k units which are sampled according to a softmax distribution over
    the activations.

    :param x:
      Current activity of each unit, optionally batched along the 0th dimension.

    :param k:
      The activity of the top k units will be allowed to remain, the rest are
      set to zero.

    :param temperature:
      Temperature to use when computing the softmax distribution over activations.
      Higher temperatures increases the entropy, lower temperatures decrease entropy.

    :param relu:
      Whether to simulate the effect of applying ReLU before KWinners

    :param inplace:
      Whether to modify x in place

    :return:
      A tensor representing the activity of x after k-winner take all.
    """
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
    """
    A stochastic K-winner take all function for creating Conv2d layers with sparse output.
    Keeps only k units which are sampled according to a softmax distribution over
    the activations.

    :param x:
      Current activity of each unit, optionally batched along the 0th dimension.

    :param k:
      The activity of the top k units will be allowed to remain, the rest are
      set to zero.

    :param temperature:
      Temperature to use when computing the softmax distribution over activations.
      Higher temperatures increases the entropy, lower temperatures decrease entropy.

    :param relu:
      Whether to simulate the effect of applying ReLU before KWinners

    :param inplace:
      Whether to modify x in place

    :return:
      A tensor representing the activity of x after k-winner take all.
    """
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