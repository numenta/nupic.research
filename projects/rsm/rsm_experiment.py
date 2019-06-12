import random
import os
import time
import sys
import json
from functools import reduce

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from rsm import RSMLayer
from rsm_samplers import MNISTSequenceSampler, mnist_pred_sequence_collate

DATA_DIR = '~/nta/datasets'
RESULTS_DIR = '/Users/jgordon/nta/results'

# writer = SummaryWriter(logdir=RESULTS_DIR + "/RSM")
writer = None


class RSMExperiment(object):
    """
    Generic class for creating tiny RSM models. This can be used with Ray
    tune or PyExperimentSuite, to run a single trial or repetition of a
    network.
    """

    def __init__(self, config):
        self.data_dir = config.get("data_dir", "data")
        self.model_filename = config.get("model_filename", "model.pth")
        self.iterations = config.get("iterations", 200)

        # Training / testing parameters
        self.batch_size = config.get("batch_size", 128)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)

        # Data parameters
        self.input_size = config.get("input_size", (1, 28, 28))
        self.sequences = config.get("sequences", "[[0, 1, 2, 3]]")

        self.learning_rate = config.get("learning_rate", 0.1)
        self.momentum = config.get("momentum", 0.9)

        self.m_groups = config.get("m_groups", 200)
        self.n_cells_per_groups = config.get("n_cells_per_groups", 6)
        self.k_winners = config.get("k_winners", 25)
        self.gamma = config.get("gamma", 0.5)
        self.eps = config.get("eps", 0.5)

        self.loss_function = nn.functional.mse_loss
        # self.lr_step_schedule = config.get("lr_step_schedule", None)

    def model_setup(self, config):
        seed = config.get("seed", random.randint(0, 10000))
        if torch.cuda.is_available():
            print("setup: Using cuda")
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
        else:
            print("setup: Using cpu")
            self.device = torch.device("cpu")

        # Build model and optimizer
        self.d_in = reduce(lambda x, y: x * y, self.input_size)
        self.model = RSMLayer(D_in=self.d_in, m=self.m_groups,
                              n=self.n_cells_per_groups,
                              k=self.k_winners,
                              gamma=self.gamma,
                              eps=self.eps)
        self.model.to(self.device)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)

        # Build sampler / data loader
        self.dataset = datasets.MNIST(DATA_DIR, download=True,
                                      train=True, transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]),)

        # Extra element for sequential prediction labels
        pred_batch_size = self.batch_size + 1
        self.sampler = MNISTSequenceSampler(self.dataset, sequences=self.sequences)
        self.train_loader = DataLoader(self.dataset,
                                       batch_size=pred_batch_size,
                                       sampler=self.sampler,
                                       collate_fn=mnist_pred_sequence_collate)

    def _view_batch(self, image_batch):
        return image_batch.reshape(self.batch_size, 1, 28, 28)

    def train_epoch(self, epoch):
        """This should be called to do one epoch of training and testing.

        Returns:
            A dict that describes progress of this epoch.
            The dict includes the key 'stop'. If set to one, this network
            should be stopped early. Training is not progressing well enough.
        """
        t1 = time.time()

        ret = {}

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):

            # Shifted input/label sequence (generate image of next item)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            x_a_pred = self.model(inputs)

            loss = self.criterion(x_a_pred, targets.reshape(self.batch_size, self.d_in))

            # Try to update board images on batch 10, each 5 epochs
            if epoch % 5 == 0 and batch_idx == 10:
                pred_grid = vutils.make_grid(self._view_batch(x_a_pred),
                                             normalize=True, scale_each=True)
                input_grid = vutils.make_grid(self._view_batch(inputs),
                                              normalize=True, scale_each=True)
                ret['input'] = input_grid
                ret['prediction'] = pred_grid

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_time = time.time() - t1

        ret["stop"] = 0

        ret["epoch_time_train"] = train_time
        ret["epoch_time"] = time.time() - t1
        ret["learning_rate"] = self.learning_rate
        print(epoch, ret)
        return ret

    def model_save(self, checkpoint_dir):
        """Save the model in this directory.

        :param checkpoint_dir:

        :return: str: The return value is expected to be the checkpoint path that
        can be later passed to `model_restore()`.
        """
        # checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        # torch.save(self.model.state_dict(), checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, self.model_filename)

        # Use the slow method if filename ends with .pt
        if checkpoint_path.endswith(".pt"):
            torch.save(self.model, checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)

        return checkpoint_path

    def model_restore(self, checkpoint_path):
        """
        :param checkpoint_path: Loads model from this checkpoint path.
        If path is a directory, will append the parameter model_filename
        """
        print("Loading from", checkpoint_path)
        if os.path.isdir(checkpoint_path):
            checkpoint_file = os.path.join(checkpoint_path, self.model_filename)
        else:
            checkpoint_file = checkpoint_path

        # Use the slow method if filename ends with .pt
        if checkpoint_file.endswith(".pt"):
            self.model = torch.load(checkpoint_file, map_location=self.device)
        else:
            self.model.load_state_dict(
                torch.load(checkpoint_file, map_location=self.device)
            )


if __name__ == '__main__':
    print("Using torch version", torch.__version__)
    print("Torch device count=%d" % torch.cuda.device_count())

    exp = RSMExperiment()
    exp.model_setup()
    for epoch in range(50):
        exp.train_epoch(epoch)
