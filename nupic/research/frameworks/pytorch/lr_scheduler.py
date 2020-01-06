#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
from bisect import bisect

from torch.optim.lr_scheduler import _LRScheduler


class ScaledLR(_LRScheduler):
    """
    Multiply the learning rate of each parameter group  by a specific factor
    assigned to the epoch. This LR scheduler could be chained together with
    other schedulers. This is useful when scaling the LR to the batch size.

    .. seealso:: See https://arxiv.org/pdf/1706.02677.pdf

    :param optimizer: Wrapped optimizer
    :param lr_scale: dict mapping initial epoch to LR scale
    :param last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, lr_scale, last_epoch=-1):
        self.lr_scale = lr_scale
        self.epochs = sorted(self.lr_scale.keys())
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        scale = self.lr_scale[self.epochs[bisect(self.epochs, self.last_epoch) - 1]]
        return map(lambda group: group["lr"] * scale, self.optimizer.param_groups)
