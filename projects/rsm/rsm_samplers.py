from itertools import chain
import random

import torch
from torch.utils.data import Sampler, BatchSampler


class PredictiveBatchSampler(BatchSampler):
    """
    Subclass of BatchSampler that avoids skipping an item at the end of each batch
    full_batch_size is sl x bs
    """

    def __init__(self, sampler, batch_size):
        super(PredictiveBatchSampler, self).__init__(sampler, batch_size, True)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size + 1:
                yield batch
                batch = [idx]  # Start next batch with extra item from last

        if len(batch) > 0 and not self.drop_last:
            yield batch


class MNISTSequenceSampler(Sampler):
    """
    Loop through one or more sequences of digits
    Draw each digit image (based on label specified by sequence) randomly
    """

    def __init__(self, data_source, sequences=None, batch_size=64, 
                 randomize_sequences=False, random_mnist_images=True,
                 use_mnist_pct=1.0):
        super(MNISTSequenceSampler, self).__init__(data_source)
        self.data_source = data_source
        # self.bsz = batch_size
        self.randomize_sequences = randomize_sequences
        self.random_mnist_images = random_mnist_images
        self.label_indices = {}  # Digit -> Indices in dataset
        self.label_cursors = {}  # Digit -> Cursor across images for each digit

        self.sequences = sequences
        self.sequence_id = 0  # Of list of sequences passed
        self.sequence_cursor = 0  # Iterates through each sequence
        self.full_sequence = list(chain.from_iterable(self.sequences))
        self.seq_length = len(self.full_sequence)
        self.use_mnist_pct = use_mnist_pct

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

    def _random_sequence_id(self):
        return random.randint(0, len(self.sequences) - 1)

    def _get_next_digit(self):
        """
        Return next integer digit in full sequence
        """
        current_seq = self.sequences[self.sequence_id]
        digit = current_seq[self.sequence_cursor]
        self.sequence_cursor += 1
        if self.sequence_cursor > len(current_seq) - 1:
            self.sequence_cursor = 0
            if self.randomize_sequences:
                self.sequence_id = self._random_sequence_id()
            else:
                self.sequence_id += 1
        if self.sequence_id > len(self.sequences) - 1:
            self.sequence_id = 0
        return digit

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
        # Ensure we start at beg of sequence on each enumerate (e.g. epoch)
        self.sequence_cursor = 0   
        while True:
            digit = self._get_next_digit()
            next_sample_id = self._get_sample_image(digit)
            yield next_sample_id
        return

    def __len__(self):
        return len(self.data_source)


def pred_sequence_collate(batch, bsz=4, seq_length=3, return_inputs=False):
    """
    Batch returned from sampler is a list of (image, label) tuples
    Offset target batch by 1 to get next image predictions

    Returns a 3-tuple which is iterated by the loader:
        data (sl x bs x pixels)
        target (sl x bs x pixels)
        pred_target (sl x bs x 1)  Label (digit)
    """
    # Transposes needed since batches come in sequential order of size sl * bs
    data = torch.stack([s[0] for s in batch[:-1]]).view(bsz, seq_length, -1).transpose(0, 1).contiguous()
    target = torch.stack([s[0] for s in batch[1:]]).view(bsz, seq_length, -1).transpose(0, 1).contiguous()
    pred_target = torch.tensor([s[1] for s in batch[1:]]).view(bsz, seq_length).t()
    input_labels = None
    if return_inputs:
        input_labels = torch.tensor([s[1] for s in batch[:-1]]).view(bsz, seq_length).t()
    return (data, target, pred_target, input_labels)


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

