# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
import unittest

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from rsm_samplers import (
    MNISTSequenceSampler, 
    MNISTBufferedDataset, 
    pred_sequence_collate
)


class TestContext(object):
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, x):
        self.saved_tensors = (x,)


class StochasticMNISTTest(unittest.TestCase):
    """
    Test that sampling classes produce correct batches of input and target images
    according to the provided sequences. 
    """

    def setUp(self):
        # Create sampler, batch sampler and data loader with collate function

        self.BSZ = 2
        self.SEQ = [[0, 1, 2, 3], [0, 3, 2, 4]]

        self.dataset = MNISTBufferedDataset("~/nta/datasets", download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]),)

        self.random_sampler = MNISTSequenceSampler(self.dataset, sequences=self.SEQ, 
                                                   batch_size=self.BSZ, random_mnist_images=True,
                                                   randomize_sequence_cursors=False)
        self.fixed_sampler = MNISTSequenceSampler(self.dataset, sequences=self.SEQ, 
                                                  batch_size=self.BSZ, random_mnist_images=False,
                                                  randomize_sequence_cursors=False)

        self.collate_fn = pred_sequence_collate

        self.random_digit_loader = DataLoader(self.dataset,
                                              batch_sampler=self.random_sampler,
                                              collate_fn=self.collate_fn)

        self.fixed_digit_loader = DataLoader(self.dataset,
                                             batch_sampler=self.fixed_sampler,
                                             collate_fn=self.collate_fn)

    def test_one(self):
        all_input_labels = []
        all_target_labels = []
        all_inputs = []

        for i in range(1):
            for j, batch in enumerate(self.random_digit_loader):
                inputs, targets, target_labels, input_labels = batch

                all_inputs.append(inputs)
                all_input_labels.append(input_labels)
                all_target_labels.append(target_labels)

                # Target and target labels have matching batch size
                self.assertEqual(targets.size(0), target_labels.size(0))

                # Input and input labels have matching batch size
                self.assertEqual(inputs.size(0), input_labels.size(0))

                self.assertEqual(inputs.size(0), self.BSZ)

                if len(all_input_labels) == 8:
                    break

        all_input_labels = torch.stack(all_input_labels, dim=0)
        all_target_labels = torch.stack(all_target_labels, dim=0)

        # Each sequence in each batch should match one of the two original sequences
        seq1col1 = all_input_labels[:4, 0]
        seq1col2 = all_input_labels[:4, 1]
        seq2col1 = all_input_labels[4:, 0]
        seq2col2 = all_input_labels[4:, 1]
        for seq in [seq1col1, seq1col2, seq2col1, seq2col2]:
            self.assertTrue(list(seq) == self.SEQ[0] or list(seq) == self.SEQ[1])

        # First 3 digit of target sequences match last 3 digits of actual sequences
        seq1col1 = all_target_labels[:3, 0]
        seq1col2 = all_target_labels[:3, 1]
        seq2col1 = all_target_labels[4:-1, 0]
        seq2col2 = all_target_labels[4:-1, 1]
        for seq in [seq1col1, seq1col2, seq2col1, seq2col2]:
            self.assertTrue(list(seq) == self.SEQ[0][1:] or list(seq) == self.SEQ[1][1:])

        # Digit images with same label are different
        if j == 0:
            zero_image_b0 = all_inputs[0, 0]
            zero_image_b1 = all_inputs[0, 1]
            self.assertTrue(zero_image_b0.sum() != zero_image_b1.sum())

    def test_fixed_digit_sampling(self):
        batch = next(iter(self.fixed_digit_loader))
        inputs, targets, target_labels, input_labels = batch

        # Digit images with same label are the same (same sum)
        zero_image_b0 = inputs[0][0]
        zero_image_b1 = inputs[0][1]
        self.assertTrue(zero_image_b0.sum() == zero_image_b1.sum())

    def test_digit_balance(self):

        random_seq_sampler = MNISTSequenceSampler(self.dataset, sequences=self.SEQ, 
                                                  batch_size=self.BSZ,
                                                  random_mnist_images=True)

        loader = DataLoader(self.dataset,
                            batch_sampler=random_seq_sampler,
                            collate_fn=self.collate_fn)

        all_inputs = []
        for i, batch in enumerate(loader):
            inputs, targets, target_labels, input_labels = batch
            all_inputs.append(input_labels)

            if i > 400:
                break

        counts = torch.stack(all_inputs).flatten().bincount()
        n_zeros = counts[0].item()
        n_ones = counts[1].item()
        # Frequency of 0s and 1s should approximately match statistics of sequences (2:1)
        self.assertAlmostEqual(n_zeros / n_ones, 2.0, delta=0.4)


if __name__ == "__main__":
    unittest.main()
