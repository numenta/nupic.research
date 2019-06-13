import random
import os
import time
import sys
from functools import reduce

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
import torchvision.utils as vutils

from rsm import RSMLayer
from rsm_samplers import MNISTSequenceSampler, mnist_pred_sequence_collate


class RSMExperiment(object):
    """
    Generic class for creating tiny RSM models. This can be used with Ray
    tune or PyExperimentSuite, to run a single trial or repetition of a
    network.
    """

    def __init__(self, config=None):
        self.data_dir = config.get("data_dir", "data")
        self.path = config.get("path", "results")
        self.model_filename = config.get("model_filename", "model.pth")
        self.graph_filename = config.get("graph_filename", "rsm.onnx")
        self.save_onnx_graph_at_checkpoint = config.get("save_onnx_graph_at_checkpoint", False)
        self.exp_name = config.get("name", "exp")
        self.writer = None

        self.iterations = config.get("iterations", 200)

        # Training / testing parameters
        self.batch_size = config.get("batch_size", 128)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)

        # Data parameters
        self.input_size = config.get("input_size", (1, 28, 28))
        self.sequences = config.get("sequences", [[0, 1, 2, 3]])

        self.learning_rate = config.get("learning_rate", 0.1)
        self.momentum = config.get("momentum", 0.9)
        self.optimizer_type = config.get("optimizer", "adam")

        self.m_groups = config.get("m_groups", 200)
        self.n_cells_per_groups = config.get("n_cells_per_groups", 6)
        self.k_winners = config.get("k_winners", 25)
        self.gamma = config.get("gamma", 0.5)
        self.eps = config.get("eps", 0.5)

        self.loss_function = nn.functional.mse_loss
        self.lr_step_schedule = config.get("lr_step_schedule", None)
        self.learning_rate_gamma = config.get("learning_rate_gamma", 0.1)

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
        self.model = RSMLayer(D_in=self.d_in, 
                              m=self.m_groups,
                              n=self.n_cells_per_groups,
                              k=self.k_winners,
                              gamma=self.gamma,
                              eps=self.eps)
        self.model.to(self.device)

        self.criterion = torch.nn.MSELoss(reduction='mean')
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.learning_rate,
                                             momentum=self.momentum)

        # Build sampler / data loader
        self.dataset = datasets.MNIST(self.data_dir, download=True,
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

    def _image_grid(self, image_batch):
        batch = image_batch.reshape(self.batch_size, 1, 28, 28)
        # make_grid returns 3 channels? -- mean
        grid = vutils.make_grid(batch, normalize=True, padding=5).mean(dim=0)  
        return grid

    def _adjust_learning_rate(self, optimizer, epoch):
        if self.lr_step_schedule is not None:
            if epoch in self.lr_step_schedule:
                self.learning_rate *= self.learning_rate_gamma
                print("Reducing learning rate to:", self.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

    def train_epoch(self, epoch):
        """This should be called to do one epoch of training and testing.

        Returns:
            A dict that describes progress of this epoch.
            The dict includes the key 'stop'. If set to one, this network
            should be stopped early. Training is not progressing well enough.
        """
        t1 = time.time()

        ret = {}
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):

            # Shifted input/label sequence (generate image of next item)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            x_a_pred, x_b, phi, psi = self.model(inputs)

            loss = self.criterion(x_a_pred, targets.reshape(self.batch_size, self.d_in))
            total_loss += loss.cpu().item()

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            loss.detach()  # Possibly needed to reduce memory reqs
            self.optimizer.step()

            # Try to update board images on batch 10, each 5 epochs
            # input_batch = self._view_batch(inputs)
            # pred_batch = self._view_batch(x_a_pred)
            if epoch % 5 == 0 and batch_idx == 10:
                ret['img_inputs'] = self._image_grid(inputs).cpu()
                ret['img_preds'] = self._image_grid(x_a_pred).cpu()
                ret['hist_w_a'] = self.model.linear_a.weight.cpu()
                ret['hist_w_b'] = self.model.linear_b.weight.cpu()
                ret['hist_w_d'] = self.model.linear_d.weight.cpu()

            if batch_idx >= self.batches_in_epoch:
                break

        train_time = time.time() - t1
        self._post_epoch(epoch)

        ret["stop"] = 0

        ret['loss'] = total_loss
        ret["epoch_time_train"] = train_time
        ret["epoch_time"] = time.time() - t1
        ret["learning_rate"] = self.learning_rate
        print(epoch, ret['loss'])
        return ret

    def _post_epoch(self, epoch):
        """
        The set of actions to do after each epoch of training: adjust learning
        rate, rezero sparse weights, and update boost strengths.
        """
        self._adjust_learning_rate(self.optimizer, epoch)

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

        if self.save_onnx_graph_at_checkpoint:
            dummy_input = (torch.rand(1, 1, 28, 28),)
            torch.onnx.export(self.model, dummy_input, self.graph_filename, 
                              verbose=True)

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

    def model_cleanup(self):
        if self.writer:
            self.writer.close()


if __name__ == '__main__':
    print("Using torch version", torch.__version__)
    print("Torch device count=%d" % torch.cuda.device_count())

    config = {
        'data_dir': os.path.expanduser('~/nta/datasets'),
        'path': os.path.expanduser('~/nta/results')
    }

    exp = RSMExperiment(config)
    exp.model_setup(config)
    for epoch in range(2):
        exp.train_epoch(epoch)
