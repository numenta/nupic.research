# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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


def compare_models(
    model1,
    model2,
    input_shape,
    epsilon=0.0001,
    num_samples=16
):
    """
    Determines whether two models are functionally equivalent or not.
    This is done by feeding in num_samples random inputs and seeing whether
    the largest output difference is within epsilon.

    :param model1: A torch.nn.Module
    :param model2: A torch.nn.Module
    :param input_shape: The expected shape of inputs, e.g. (28,28) for MNIST
    :param epsilon: Tolerance
    :param num_samples: Number of random samples to test

    :return: Boolean
    """
    x = torch.randn((num_samples,) + input_shape)

    with torch.no_grad():
        y1 = model1(x)
        y2 = model2(x)
        diff = (y1 - y2).abs()

    return diff.max() <= epsilon
