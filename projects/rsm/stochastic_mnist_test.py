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
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rsm_samplers import MNISTSequenceSampler, PredictiveBatchSampler, pred_sequence_collate


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
        self.SL = 8

        self.dataset = datasets.MNIST("~/nta/datasets", download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]),)

        random_sampler = MNISTSequenceSampler(self.dataset, sequences=self.SEQ, randomize_sequences=False, random_mnist_images=True)
        self.random_batch_sampler = PredictiveBatchSampler(random_sampler, batch_size=self.SL * self.BSZ)
        fixed_sampler = MNISTSequenceSampler(self.dataset, sequences=self.SEQ, randomize_sequences=False, random_mnist_images=False)
        self.fixed_batch_sampler = PredictiveBatchSampler(fixed_sampler, batch_size=self.SL * self.BSZ)

        self.collate_fn = partial(pred_sequence_collate, 
                                  bsz=self.BSZ,
                                  seq_length=self.SL,
                                  return_inputs=True)

        self.random_digit_loader = DataLoader(self.dataset,
                                              batch_sampler=self.random_batch_sampler,
                                              collate_fn=self.collate_fn)

        self.fixed_digit_loader = DataLoader(self.dataset,
                                             batch_sampler=self.fixed_batch_sampler,
                                             collate_fn=self.collate_fn)

    def test_one(self):
        # Each loop of both epoch (i) and batch (j) should begin at the start of SEQ
        for i in range(2):
            for j, batch in enumerate(self.random_digit_loader):
                inputs, targets, target_labels, input_labels = batch

                expected_input_labels = torch.tensor([
                    [0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3],
                    [0, 0],
                    [3, 3],
                    [2, 2],
                    [4, 4],
                ])
                self.assertEqual((expected_input_labels == input_labels).sum(), self.BSZ * self.SL)
                self.assertEqual(input_labels.size(), expected_input_labels.size())

                # Target labels should be next in sequence at each location
                expected_target_labels = torch.tensor([
                    [1, 1],
                    [2, 2],
                    [3, 3],
                    [0, 0],
                    [3, 3],
                    [2, 2],
                    [4, 4],
                    [0, 0]
                ])
                self.assertEqual((expected_target_labels == target_labels).sum(), self.BSZ * self.SL)
                self.assertEqual(target_labels.size(), expected_target_labels.size())

                # Target and target labels have matching batch size and sequence length
                self.assertEqual(targets.size(0), target_labels.size(0))
                self.assertEqual(targets.size(1), target_labels.size(1))

                # Input and input labels have matching batch size and sequence length
                self.assertEqual(inputs.size(0), input_labels.size(0))
                self.assertEqual(inputs.size(1), input_labels.size(1))

                # Digit images with same label are different
                zero_image_b0 = inputs[0][0]
                zero_image_b1 = inputs[0][1]
                self.assertTrue(zero_image_b0.sum() != zero_image_b1.sum())

                if j > 2:
                    break

    def test_fixed_digit_sampling(self):
        batch = next(iter(self.fixed_digit_loader))
        inputs, targets, target_labels, input_labels = batch

        # Digit images with same label are the same (same sum)
        zero_image_b0 = inputs[0][0]
        zero_image_b1 = inputs[0][1]
        self.assertTrue(zero_image_b0.sum() == zero_image_b1.sum())

    def test_digit_balance(self):

        random_seq_sampler = MNISTSequenceSampler(self.dataset, sequences=self.SEQ, randomize_sequences=True, random_mnist_images=True)
        random_seq_batch_sampler = PredictiveBatchSampler(random_seq_sampler, batch_size=self.SL * self.BSZ)

        loader = DataLoader(self.dataset,
                            batch_sampler=random_seq_batch_sampler,
                            collate_fn=self.collate_fn)

        all_inputs = []
        for i, batch in enumerate(loader):
            inputs, targets, target_labels, input_labels = batch
            all_inputs.append(input_labels)

            if i > 100:
                break

        counts = torch.stack(all_inputs).flatten().bincount()
        n_zeros = counts[0].item()
        n_ones = counts[1].item()
        # Frequency of 0s and 1s should approximately match statistics of sequences (2:1)
        self.assertAlmostEqual(n_zeros / n_ones, 2.0, delta=0.4)


if __name__ == "__main__":
    unittest.main()
