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


def mnist_pred_sequence_collate(batch):
    # Predictive auto-encoder, targets are next image in sequence
    inputs = [item[0] for item in batch[:-1]]
    targets = [item[0] for item in batch[1:]]
    return [torch.stack(inputs, 0), torch.stack(targets, 0)]