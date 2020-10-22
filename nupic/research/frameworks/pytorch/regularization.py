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

import torch


@torch.no_grad()
def l1_regularization_step(params, lr, weight_decay=1e-3):
    """
    Performs L1 regularization gradient step in place


    Example usage:

        # Assume `model` is an instance of a torch.nn.Module subclass, and `optimizer`
        # is used to perform SGD updates on the parameters of `model`

        # The following lines perform gradient updates on a specified loss with an L1
        # penalty term

        # Note that the L1 updates are performed separately from the updates on the
        # regular objective function

        loss.backward()
        optimizer.step()
        l1_regularization_step(params=model.parameters(), lr=0.1, weight_decay=1e-3)


    :param params: a list of parameters on which the L1 regularization update will be
                   performed, conditioned on whether attribute `requires_grad` is True
    :param lr: the learning rate used during optimization, analogous to the `lr`
               parameter in `torch.optim.SGD`
    :param weight_decay: the L1 penalty coefficient, analogous to the `weight_decay`
                         parameter (used as the L2 penalty coefficient) in
                         `torch.optim.SGD`
    """
    for p in params:
        if p.requires_grad:
            grad = torch.sign(p.data)
            p.add_(grad, alpha=-lr * weight_decay)
