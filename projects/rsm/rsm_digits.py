import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from rsm import RSMLayer

DATA_DIR = '~/nta/datasets'
RESULTS_DIR = '/Users/jgordon/nta/results'

writer = SummaryWriter(logdir=RESULTS_DIR + "/RSM")


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
            next_sample = self.label_indices[digit][cursor]
            self.label_cursors[digit] += 1
            yield next_sample
        return

    def __len__(self):
        return len(self.data_source)


class MNISTSequenceExperiment(object):

    def __init__(self, batch_size=200, d_in=28 * 28, sequences=None):
        super(MNISTSequenceExperiment, self).__init__()
        self.batch_size = batch_size
        self.d_in = d_in
        self.sequences = sequences
        self.setup()

    def setup(self):
        self.model = RSMLayer(self.d_in)
        # writer.add_graph(self.model)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        self.dataset = datasets.MNIST(DATA_DIR, download=True,
                                      train=True, transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]),)
        # Extra element for sequential prediction labels
        pred_batch_size = self.batch_size + 1
        self.sampler = MNISTSequenceSampler(
            self.dataset, sequences=self.sequences)
        self.loader = DataLoader(self.dataset,
                                 batch_size=pred_batch_size,
                                 sampler=self.sampler)

    def _view_batch(self, image_batch):
        return image_batch.reshape(self.batch_size, 1, 28, 28)

    def run(self):
        for epoch, (raw_inputs, raw_labels) in enumerate(self.loader):

            # Shifted input/label sequence (generate image of next item)
            inputs = raw_inputs[:-1]
            labels = raw_inputs[1:]
            x_a_pred = self.model(inputs)

            loss = self.criterion(x_a_pred, labels.reshape(self.batch_size, self.d_in))
            print(epoch, loss.item())

            if epoch % 10 == 0:
                writer.add_scalar('loss', loss, epoch)  
                pred_grid = vutils.make_grid(self._view_batch(x_a_pred), normalize=True, scale_each=True)
                input_grid = vutils.make_grid(self._view_batch(inputs), normalize=True, scale_each=True)
                writer.add_image('input', input_grid, epoch)
                writer.add_image('prediction', pred_grid, epoch)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch += 1


SEQUENCES = [
    [0, 1, 2, 3]
]

exp = MNISTSequenceExperiment(sequences=SEQUENCES)

exp.run()
