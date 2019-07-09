from itertools import chain
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, DataLoader


class MNISTSequenceLoader(DataLoader):
    """
    Loop through one or more sequences of digits
    Draw each digit image (based on label specified by sequence) randomly
    """

    def __init__(self, data_source, sequences=None, batch_size=64, 
                 randomize_sequences=False, random_mnist_images=True,
                 use_mnist_pct=1.0):
        super(MNISTSequenceLoader, self).__init__(data_source, batch_size=batch_size)
        self.data_source = data_source
        self.randomize_sequences = randomize_sequences
        self.random_mnist_images = random_mnist_images
        self.use_mnist_pct = use_mnist_pct
        self.label_indices = {}  # Digit -> Indices in dataset
        self.label_cursors = {}  # Digit -> Cursor across images for each digit

        self.sequences = sequences
        self.n_sequences = len(self.sequences)

        # Each of these stores both current and next batch state (2 x batch_size)
        self.sequence_id = torch.stack((self._next_sequence_ids(), self._next_sequence_ids()))  # Iterate over subsequences
        self.sequence_cursor = torch.stack((torch.zeros(batch_size).long(), torch.ones(batch_size).long()))  # Iterates over sequence items

        self.seq_lengths = torch.tensor([len(subseq) for subseq in self.sequences])
        self.sequences_mat = pad_sequence(torch.tensor(self.sequences), batch_first=True, padding_value=0)

        # Get index for each digit (that appears in a passed sequence)
        for seq in sequences:
            for digit in seq:
                if digit not in self.label_indices:
                    mask = (data_source.targets == digit).nonzero().flatten()
                    idx = torch.randperm(mask.size(0))
                    if self.use_mnist_pct < 1.0:
                        idx = idx[:int(self.use_mnist_pct * len(idx))]
                    self.label_indices[digit] = mask[idx]
                    self.label_cursors[digit] = 0

    def _next_sequence_ids(self):
        return torch.LongTensor(self.batch_size).random_(0, self.n_sequences)

    def _get_next_batch(self):
        """
        """
        # First row is current inputs
        inp_labels_batch = self.sequences_mat[self.sequence_id[0], self.sequence_cursor[0]]
        img_idxs = [self._get_sample_image(digit.item()) for digit in inp_labels_batch]
        inp_images_batch = self.data_source.data[img_idxs].float().view(self.batch_size, -1)

        # Second row is next (predicted) inputs
        tgt_labels_batch = self.sequences_mat[self.sequence_id[1], self.sequence_cursor[1]]
        img_idxs = [self._get_sample_image(digit.item()) for digit in inp_labels_batch]
        tgt_images_batch = self.data_source.data[img_idxs].float().view(self.batch_size, -1)

        # Roll next to current
        self.sequence_id[0] = self.sequence_id[1]
        self.sequence_cursor[0] = self.sequence_cursor[1]

        # Increment cursors and select new random subsequences for those that have terminated
        self.sequence_cursor[1] += 1
        roll_mask = self.sequence_cursor[1] >= self.seq_lengths[self.sequence_id[1]]

        if roll_mask.sum() > 0:
            # Roll items to 0 of randomly chosen next subsequence
            self.sequence_id[1, roll_mask] = torch.LongTensor(len(roll_mask)).random_(0, self.n_sequences)
            self.sequence_cursor[1, roll_mask] = 0

        return inp_images_batch, tgt_images_batch, tgt_labels_batch, inp_labels_batch

    def _get_sample_image(self, digit):
        """
        Return a sample image id for digit from MNIST
        """
        cursor = self.label_cursors[digit]
        if self.random_mnist_images:
            # If not random, always take first digit
            self.label_cursors[digit] += 1
        indices = self.label_indices[digit]
        if cursor >= len(indices) - 1:
            # Begin sequence from beginning -- should we re-shuffle?
            self.label_cursors[digit] = cursor = 0
        return indices[cursor].item()

    def __iter__(self):
        while True:
            yield self._get_next_batch()
        return

    def __len__(self):
        return len(self.data_source)


def pred_sequence_collate(batch):
    """
    """
    print(batch.size())
    return batch


class PTBSequenceSampler(Sampler):
    """
    """

    def __init__(self, data_source, batch_size=64, seq_length=1):
        super(PTBSequenceSampler, self).__init__(data_source)
        self.bsz = batch_size
        self.seq_length = seq_length
        self.data_source = data_source
        self.data_len = len(self.data_source)
        # Choose initial random offsets into PTB, one per item in batch
        batch_offsets = (torch.rand(self.bsz) * (self.data_len - 1)).long()
        # Increment 1 word ID each row/batch
        inc = torch.arange(0, self.seq_length, dtype=torch.long).expand((self.bsz, self.seq_length)).t()
        self.batch_idxs = batch_offsets.expand((self.seq_length, self.bsz)) + inc

    def __iter__(self):
        # Yield a single batch of (seq_len x batch_size) words, each at a different offset into PTB
        while True:
            data = self.data_source[self.batch_idxs]
            target = self.data_source[self.batch_idxs + 1]
            self.batch_idxs += self.seq_length  # Move down to next sequence (maybe 1 row)
            self.batch_idxs[self.batch_idxs > (self.data_len - 2)] = 0  # Wrap to start
            yield data, target
        return

    def __len__(self):
        return len(self.data_source) // self.bsz


def vector_batch(word_ids, vector_dict):
    vectors = []
    for word_id in word_ids.flatten():
        vectors.append(vector_dict[word_id.item()])
    return torch.stack(vectors).view(word_ids.size(0), word_ids.size(1), -1)


def ptb_pred_sequence_collate(batch, vector_dict=None):
    """
    Return minibatches, shape (seq_len, batch_size, embed dim)
    """
    data, target = batch
    data = vector_batch(data, vector_dict)
    pred_target = target
    target = vector_batch(target, vector_dict)
    return (data, target, pred_target)

