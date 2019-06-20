import torch
from torch.utils.data import Sampler


class MNISTSequenceSampler(Sampler):
    """
    Loop through one or more sequences of digits
    Draw each digit image (based on label specified by sequence) randomly

    Currently deterministic sequential

    TODO: Option to randomize sequence order?
    """

    def __init__(self, data_source, sequences=None):
        super(MNISTSequenceSampler, self).__init__(data_source)
        self.data_source = data_source
        self.label_indices = {}  # Digit -> Indices in dataset
        self.label_cursors = {}  # Digit -> Random sequence cursor

        self.sequences = sequences
        self.sequence_id = 0  # Of list of sequences passed
        self.sequence_cursor = -1  # Iterates through each sequence

        # Get index for each digit (that appears in a passed sequence)
        for seq in sequences:
            for digit in seq:
                if digit not in self.label_indices:
                    mask = (data_source.targets == digit).nonzero().flatten()
                    idx = torch.randperm(mask.size(0))
                    self.label_indices[digit] = mask[idx]
                    self.label_cursors[digit] = 0

    def _get_next_digit(self):
        current_seq = self.sequences[self.sequence_id]
        self.sequence_cursor += 1
        if self.sequence_cursor >= len(current_seq) - 1:
            self.sequence_cursor = -1
            self.sequence_id += 1
        if self.sequence_id >= len(self.sequences) - 1:
            self.sequence_id = 0
            current_seq = self.sequences[self.sequence_id]
        digit = current_seq[self.sequence_cursor]
        return digit

    def __iter__(self):
        # Cleaner?
        while True:
            digit = self._get_next_digit()
            cursor = self.label_cursors[digit]
            indices = self.label_indices[digit]
            if cursor >= len(indices) - 1:
                self.label_cursors[digit] = cursor = 0
            next_sample = indices[cursor]
            self.label_cursors[digit] += 1
            yield next_sample
        return

    def __len__(self):
        return len(self.data_source)


def pred_sequence_collate(batch):
    # Predictive auto-encoder, targets are next image in sequence
    inputs = [item[0] for item in batch[:-1]]
    targets = [item[0] for item in batch[1:]]
    pred_targets = [item[1] for item in batch[1:]]
    return [torch.stack(inputs, 0), torch.stack(targets, 0), torch.LongTensor(pred_targets)]


class LangSequenceSampler(Sampler):
    """
    """

    def __init__(self, data_source, batch_size=64, seq_length=35, parallel_seq=False):
        super(LangSequenceSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.parallel_seq = parallel_seq
        if self.parallel_seq:
            # Convert to 3-dim tensor for nn.LSTM
            self.data_source = self.batchify(self.data_source)

    def batchify(self, data):
        """
        Ref: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        """
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * self.batch_size)
        # Evenly divide the data across the self.batch_size batches.
        data = data.view(self.batch_size, -1).t().contiguous()
        return data

    def get_parallel_batch(self, i):
        seq_len = min(self.seq_length, len(self.data_source) - 1 - i)
        data = self.data_source[i:i + seq_len]
        target = self.data_source[i + 1: i + 1 + seq_len].view(-1)
        return data, target

    def __iter__(self):
        if self.parallel_seq:
            for i in range(0, self.data_source.size(0), self.seq_length):
                yield self.get_parallel_batch(i)
        else:
            n_batches = len(self)
            for i in range(n_batches):
                st = i * self.batch_size
                # Batch is batch_size + 1 to allow next word prediction
                en = st + self.batch_size + 1
                batch = self.data_source[st:en]
                yield batch
        return

    def __len__(self):
        return len(self.data_source) // self.batch_size


def language_pred_sequence_collate(batch):
    data, target = batch
    return (data, target, target)

