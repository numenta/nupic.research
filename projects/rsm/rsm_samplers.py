from itertools import chain
import random

import torch
from torch.utils.data import Sampler


class MNISTSequenceSampler(Sampler):
    """
    Loop through one or more sequences of digits
    Draw each digit image (based on label specified by sequence) randomly
    """

    def __init__(self, data_source, sequences=None, batch_size=64, randomize_sequences=False):
        super(MNISTSequenceSampler, self).__init__(data_source)
        self.data_source = data_source
        self.bsz = batch_size
        self.randomize_sequences = randomize_sequences
        self.label_indices = {}  # Digit -> Indices in dataset
        self.label_cursors = {}  # Digit -> Random sequence cursor

        self.sequences = sequences
        self.sequence_id = 0  # Of list of sequences passed
        self.sequence_cursor = -1  # Iterates through each sequence
        self.full_sequence = list(chain.from_iterable(self.sequences))
        self.seq_length = len(self.full_sequence)

        # Get index for each digit (that appears in a passed sequence)
        for seq in sequences:
            for digit in seq:
                if digit not in self.label_indices:
                    mask = (data_source.targets == digit).nonzero().flatten()
                    idx = torch.randperm(mask.size(0))
                    self.label_indices[digit] = mask[idx]
                    self.label_cursors[digit] = 0

    def _random_sequence_id(self):
        return random.randint(0, len(self.sequences) - 1)

    def _get_next_digit(self):
        """
        Return next integer digit in full sequence
        """
        current_seq = self.sequences[self.sequence_id]
        self.sequence_cursor += 1
        if self.sequence_cursor >= len(current_seq) - 1:
            self.sequence_cursor = -1
            if self.randomize_sequences:
                self.sequence_id = self._random_sequence_id()
            else:
                self.sequence_id += 1
        if self.sequence_id >= len(self.sequences) - 1:
            self.sequence_id = 0
            current_seq = self.sequences[self.sequence_id]
        digit = current_seq[self.sequence_cursor]
        return digit

    def _get_sample(self, digit):
        """
        Return a sample image id for digit from MNIST
        """
        cursor = self.label_cursors[digit]
        indices = self.label_indices[digit]
        if cursor >= len(indices) - 1:
            # Begin sequence from beginning -- should we re-shuffle?
            self.label_cursors[digit] = cursor = 0
        return indices[cursor].item()

    def _get_sequence(self):
        input_seq = []
        digit_seq = []
        for i in range(self.seq_length + 1):
            digit = self._get_next_digit()
            next_sample_id = self._get_sample(digit)
            input_seq.append(next_sample_id)
            digit_seq.append(digit)
            self.label_cursors[digit] += 1
        return input_seq, digit_seq

    def __iter__(self):
        while True:
            input_id_batch = []
            digit_batch = []
            for j in range(self.bsz):
                input_id_seq, digit_seq = self._get_sequence()
                input_id_batch.extend(input_id_seq)
                digit_batch.extend(digit_seq)
            yield input_id_batch, torch.randn(4)
        return

    def __len__(self):
        return len(self.data_source)


def pred_sequence_collate(batch):
    # Batch is a list of (image, label) tuples
    # Offset target batch by 1 to get next image predictions
    # batch = batch.view(self.seq_length, self.bsz)
    print('batch', batch)
    data = torch.stack([s[0].flatten() for s in batch[:-1]], 0)
    target = torch.stack([s[0].flatten() for s in batch[1:]], 0)
    pred_target = torch.tensor([s[1] for s in batch[1:]])
    print('data', data.size())
    print('pred_target', pred_target.size())
    # data = input_images.view(self.seq_length + 1, self.bsz, -1)
    # targets = torch.tensor(digit_batch).view(self.seq_length + 1, self.bsz)
    return (data, target, pred_target)


class LangSequenceSampler(Sampler):
    """
    """

    def __init__(self, data_source, batch_size=64, seq_length=35):
        super(LangSequenceSampler, self).__init__(data_source)
        self.bsz = batch_size
        self.seq_length = seq_length
        self.data_source = self.batchify(data_source)

    def batchify(self, data):
        """
        Ref: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        """
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // self.bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * self.bsz)
        # Evenly divide the data across the self.bsz batches.
        data = data.view(self.bsz, -1).t().contiguous()
        return data

    def __iter__(self):
        for i in range(0, self.data_source.size(0), self.seq_length):
            seq_len = min(self.seq_length, len(self.data_source) - 1 - i)
            data = self.data_source[i:i + seq_len]
            target = self.data_source[i + 1: i + 1 + seq_len].view(-1)
            yield data, target
        return

    def __len__(self):
        return len(self.data_source) // self.bsz


def language_pred_sequence_collate(batch):
    data, target = batch
    return (data, target, target)

