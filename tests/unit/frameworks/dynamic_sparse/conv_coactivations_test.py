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

import unittest

import torch

from nupic.research.frameworks.dynamic_sparse.networks import DSConv2d

# ------------------------------------------
# Helper functions for brute force method.
# ------------------------------------------


def coactivation(t1, t2, alpha, mean_activations):
    """
    :param t1: input unit
    :param t1: output unit
    :param alpha: activity threshold
    :param mean_activations: average activations of input and output
    """
    a1, a2 = alpha if hasattr(alpha, "__iter__") else (alpha, alpha)

    t1, t2 = (t1 - mean_activations[0], t2 - mean_activations[1])

    s = torch.abs(t1).gt(a1) * torch.abs(t2).gt(a2)
    return s


def get_indices_of_input_and_filter(
        n, m, in_channels, kernel_size, padding, stride):
    """
    Assumes dilation=1 and grouping=1
    """

    k1, k2 = kernel_size
    p1, p2 = padding
    s1, s2 = stride

    i1, i2 = (0, 0)

    i1 -= p1
    i2 -= p2

    i1 += n * s1
    i2 += m * s2

    indxs = []
    for c_in in range(in_channels):
        for n_k1 in range(k1):
            for m_k2 in range(k2):
                filter_indx = (c_in, n_k1, m_k2)
                input_indx = (c_in, i1 + n_k1, i2 + m_k2)
                indxs.append((input_indx, filter_indx))

    return indxs


def calc_coactivations(
    shape,
    padding,
    stride,
    input_tensor,
    output_tensor,
    mean_activations,
    alpha
):

    c_out = shape[0]
    c_in = shape[1]
    kernel_size = (shape[2], shape[3])

    b_out = output_tensor.shape[0]
    n_out = output_tensor.shape[2]
    m_out = output_tensor.shape[3]

    h = torch.zeros(shape)
    for b_ in range(b_out):
        for c_out_ in range(c_out):
            for n_out_ in range(n_out):
                for m_out_ in range(m_out):
                    unit_1 = output_tensor[b_, c_out_, n_out_, m_out_]
                    indxs = get_indices_of_input_and_filter(
                        n_out_, m_out_, c_in, kernel_size, padding, stride)

                    for input_indx, filter_indx in indxs:
                        c_in_, n_in_, m_in_ = input_indx
                        c_fl_, n_fl_, m_fl_ = filter_indx
                        unit_2 = input_tensor[b_, c_in_, n_in_, m_in_]

                        if coactivation(unit_2,
                                        unit_1, alpha, mean_activations):
                            h[c_out_, c_fl_, n_fl_, m_fl_] += 1
    return h


# ------------------
# Tests
# ------------------


class CoactivationsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_via_brute_force_comparison(self):

        batch_size = 2
        in_channels = 4
        out_channels = 4
        kernel_size = (2, 2)
        stride = (1, 1)
        padding = 0

        conv = DSConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            sparsity=0.98,
            hebbian_prune_frac=0.99,
            prune_dims=[],
            magnitude_prune_frac=0.00,
            coactivation_test="variance",
            update_nsteps=1,
        )
        conv.init_coactivation_tracking()

        input_tensor = torch.randn(batch_size, in_channels, *kernel_size)
        output_tensor = conv(input_tensor)
        mean_activations = (input_tensor.mean(), output_tensor.mean())

        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        alpha = conv.get_activity_threshold(input_tensor, output_tensor)

        coacts = calc_coactivations(
            conv.weight.shape, padding, stride,
            input_tensor, output_tensor, mean_activations, alpha)
        assert conv.coactivations.allclose(coacts, atol=0, rtol=0)
        conv.progress_connections()

        input_tensor = torch.randn(batch_size, in_channels, *kernel_size)
        output_tensor = conv(input_tensor)
        alpha = conv.get_activity_threshold(input_tensor, output_tensor)
        mean_activations = (input_tensor.mean(), output_tensor.mean())

        coacts = calc_coactivations(
            conv.weight.shape, padding, stride,
            input_tensor, output_tensor, mean_activations, alpha)
        assert conv.coactivations.allclose(coacts, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
