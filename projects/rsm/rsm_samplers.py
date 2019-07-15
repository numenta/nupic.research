from PIL import Image
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from torchvision import datasets


class MNISTBufferedDataset(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNISTBufferedDataset, self).__init__(root, train=train, transform=transform, 
                                                   target_transform=target_transform, download=download)

    def __getitem__(self, index):
        """
        Override to allow generation of white noise for index -1

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index == -1:
            # Noise
            target = np.random.randint(10)  # -1
            img = np.random.rand(28, 28)
        else:
            img, target = self.data[index].numpy(), int(self.targets[index])

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNISTSequenceSampler(Sampler):
    """
    Loop through one or more sequences of digits
    Draw each digit image (based on label specified by sequence) randomly

    TODO: Having this work with a custom DataSet that draws random 
    MNIST digits may be more appropriate
    """

    def __init__(self, data_source, sequences=None, batch_size=64, 
                 random_mnist_images=True, randomize_sequence_cursors=True,
                 max_batches=100,
                 use_mnist_pct=1.0, noise_buffer=False):
        super(MNISTSequenceSampler, self).__init__(data_source)
        self.data_source = data_source
        self.random_mnist_images = random_mnist_images
        self.randomize_sequence_cursors = randomize_sequence_cursors
        self.use_mnist_pct = use_mnist_pct
        self.noise_buffer = noise_buffer
        self.max_batches = max_batches
        self.bsz = batch_size
        self.label_indices = {}  # Digit -> Indices in dataset
        self.label_cursors = {}  # Digit -> Cursor across images for each digit

        sequences = list(sequences)  # Avoid changing underlying sequence list
        if self.noise_buffer:
            for seq in sequences:
                if seq[-1] != -1:
                    seq.append(-1)
        self.sequences = sequences
        self.n_sequences = len(self.sequences)
        self.seq_lengths = torch.tensor([len(subseq) for subseq in self.sequences])

        # Each of these stores both current and next batch state (2 x batch_size)
        self.sequence_id = torch.stack((self._init_sequence_ids(), self._init_sequence_ids()))  # Iterate over subsequences
        first_batch_cursors = self._init_sequence_cursors()
        self.sequence_cursor = torch.stack((first_batch_cursors, first_batch_cursors))  # Iterates over sequence items
        self._increment_next()

        self.sequences_mat = pad_sequence(torch.tensor(self.sequences), batch_first=True, padding_value=-99)

        # Get index for each digit (that appears in a passed sequence)
        for seq in sequences:
            for digit in seq:
                if digit != -1 and digit not in self.label_indices:
                    mask = (data_source.targets == digit).nonzero().flatten()
                    idx = torch.randperm(mask.size(0))
                    if self.use_mnist_pct < 1.0:
                        idx = idx[:int(self.use_mnist_pct * len(idx))]
                    self.label_indices[digit] = mask[idx]
                    self.label_cursors[digit] = 0

    def _init_sequence_ids(self):
        return torch.LongTensor(self.bsz).random_(0, self.n_sequences)

    def _init_sequence_cursors(self):
        if self.randomize_sequence_cursors:
            lengths = self.seq_lengths[self.sequence_id[0]]
            cursors = (torch.FloatTensor(self.bsz).uniform_(0, 1) * lengths.float()).long()
        else:
            cursors = torch.zeros(self.bsz).long()
        return cursors

    def _increment_next(self):
        # Increment cursors and select new random subsequences for those that have terminated
        self.sequence_cursor[1] += 1
        roll_mask = self.sequence_cursor[1] >= self.seq_lengths[self.sequence_id[1]]

        if roll_mask.sum() > 0:
            # Roll items to 0 of randomly chosen next subsequence
            self.sequence_id[1, roll_mask] = torch.LongTensor(1, roll_mask.sum()).random_(0, self.n_sequences)
            self.sequence_cursor[1, roll_mask] = 0

    def _get_next_batch(self):
        """
        """
        # First row is current inputs
        inp_labels_batch = self.sequences_mat[self.sequence_id[0], self.sequence_cursor[0]]
        inp_idxs = [self._get_sample_image(digit.item()) for digit in inp_labels_batch]

        # Second row is next (predicted) inputs
        tgt_labels_batch = self.sequences_mat[self.sequence_id[1], self.sequence_cursor[1]]
        tgt_idxs = [self._get_sample_image(digit.item()) for digit in tgt_labels_batch]

        # Roll next to current
        self.sequence_id[0] = self.sequence_id[1]
        self.sequence_cursor[0] = self.sequence_cursor[1]

        self._increment_next()

        # return (inp_images_batch, tgt_images_batch, tgt_labels_batch, inp_labels_batch)
        return inp_idxs + tgt_idxs

    def _get_sample_image(self, digit):
        """
        Return a sample image id for digit from MNIST
        """
        if digit == -1:
            # Generate white noise
            return -1
        else:
            cursor = self.label_cursors[digit]
            if self.random_mnist_images:
                # If not random, always take first digit
                self.label_cursors[digit] += 1
            indices = self.label_indices[digit]
            if cursor >= len(indices) - 1:
                # Begin sequence from beginning & shuffle
                self.label_cursors[digit] = cursor = 0
                idx = torch.randperm(len(self.label_indices[digit]))
                self.label_indices[digit] = indices = self.label_indices[digit][idx]
            return indices[cursor].item()

    def __iter__(self):
        for i in range(len(self)):
            yield self._get_next_batch()
        return

    def __len__(self):
        return self.max_batches


def pred_sequence_collate(batch):
    """
    """
    bsz = len(batch) // 2
    inp_tuples = batch[:bsz]
    tgt_tuples = batch[bsz:]
    inp_images_batch = torch.stack([item[0] for item in inp_tuples]).view(bsz, -1)
    tgt_images_batch = torch.stack([item[0] for item in tgt_tuples]).view(bsz, -1)
    inp_labels_batch = torch.tensor([item[1] for item in inp_tuples])
    tgt_labels_batch = torch.tensor([item[1] for item in tgt_tuples])
    return (inp_images_batch, tgt_images_batch, tgt_labels_batch, inp_labels_batch)


class PTBSequenceSampler(Sampler):
    """
    """

    def __init__(self, data_source, batch_size=64, max_batches=1000000):
        super(PTBSequenceSampler, self).__init__(None)
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.data_source = data_source
        self.data_len = len(self.data_source)
        # Choose initial random offsets into PTB, one per item in batch
        self.batch_idxs = (torch.rand(self.batch_size) * (self.data_len - 1)).long()

    def __iter__(self):
        # Yield the next single batch of (batch_size) word IDs, each at a different offset into PTB
        for i in range(len(self)):
            # yield data, target
            yield self.batch_idxs, self.batch_idxs + 1
            self.batch_idxs += 1  # Next token row
            self.batch_idxs[self.batch_idxs > (self.data_len - 2)] = 0  # Wrap to start
        return

    def __len__(self):
        return self.max_batches


def vector_batch(word_ids, vector_dict):
    vectors = []
    for word_id in word_ids:
        vectors.append(vector_dict[word_id.item()])
    return torch.stack(vectors).view(word_ids.size(0), -1)


def ptb_pred_sequence_collate(batch, vector_dict=None):
    """
    Return minibatches, shape (batch_size, embed dim)
    """
    data, target = batch
    pred_input = data
    data = vector_batch(data, vector_dict)
    pred_target = target
    target = vector_batch(target, vector_dict)
    return (data, target, pred_target, pred_input)

