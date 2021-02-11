#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import torch
from torch.distributions.multinomial import Multinomial
from torch.nn.functional import softmax


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
    A stochastic K-winner take all function for Conv2d layers with sparse output.
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
