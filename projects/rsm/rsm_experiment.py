import random
import os
import time
import sys
import math
from functools import reduce

import torch
from torchnlp.datasets import penn_treebank_dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

import lang_util
from rsm import RSMLayer
from rsm_samplers import MNISTSequenceSampler, pred_sequence_collate, language_pred_sequence_collate


class RSMExperiment(object):
    """
    Generic class for creating tiny RSM models. This can be used with Ray
    tune or PyExperimentSuite, to run a single trial or repetition of a
    network.
    """

    def __init__(self, config=None):
        self.data_dir = config.get("data_dir", "data")
        self.path = config.get("path", "results")
        self.model_filename = config.get("model_filename", "model.pt")
        self.pred_model_filename = config.get("pred_model_filename", "pred_model.pt")
        self.graph_filename = config.get("graph_filename", "rsm.onnx")
        self.save_onnx_graph_at_checkpoint = config.get("save_onnx_graph_at_checkpoint", False)
        self.exp_name = config.get("name", "exp")
        self.batch_log_interval = config.get("batch_log_interval", 0)
        self.debug = config.get("debug", False)
        self.writer = None

        self.iterations = config.get("iterations", 200)
        self.dataset_kind = config.get("dataset", "mnist")

        # Training / testing parameters
        self.batch_size = config.get("batch_size", 128)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)

        # Data parameters
        self.input_size = config.get("input_size", (1, 28, 28))
        self.sequences = config.get("sequences", [[0, 1, 2, 3]])

        self.learning_rate = config.get("learning_rate", 0.1)
        self.momentum = config.get("momentum", 0.9)
        self.optimizer_type = config.get("optimizer", "adam")

        # Model
        self.m_groups = config.get("m_groups", 200)
        self.n_cells_per_group = config.get("n_cells_per_group", 6)
        self.k_winners = config.get("k_winners", 25)
        self.gamma = config.get("gamma", 0.5)
        self.eps = config.get("eps", 0.5)
        self.k_winner_cells = config.get("k_winner_cells", 1)

        # Tweaks
        self.cell_winner_softmax = config.get("cell_winner_softmax", False)
        self.activation_fn = config.get("activation_fn", 'tanh')
        self.active_dendrites = config.get("active_dendrites", None)
        self.col_output_cells = config.get("col_output_cells", None)

        # Predictor network
        self.predictor_hidden_size = config.get("predictor_hidden_size", None)
        self.predictor_output_size = config.get("predictor_output_size", 10)

        # Embeddings for language modeling
        self.embed_dim = config.get("embed_dim", 0)
        self.vocab_size = config.get("vocab_size", 0)

        self.loss_function = config.get("loss_function", "MSELoss")
        self.lr_step_schedule = config.get("lr_step_schedule", None)
        self.learning_rate_gamma = config.get("learning_rate_gamma", 0.1)

    def _build_dataloader(self):
        # Extra element for sequential prediction labels
        pred_batch_size = self.batch_size + 1

        self.test_loader = None
        if self.dataset_kind == 'mnist':
            self.dataset = datasets.MNIST(self.data_dir, download=True,
                                          train=True, transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]),)
            self.test_dataset = datasets.MNIST(self.data_dir, download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,))
                                               ]),)

            self.sampler = MNISTSequenceSampler(self.dataset, sequences=self.sequences)
            self.train_loader = DataLoader(self.dataset,
                                           batch_size=pred_batch_size,
                                           sampler=self.sampler,
                                           collate_fn=pred_sequence_collate)
            self.test_loader = DataLoader(self.test_dataset,
                                          batch_size=pred_batch_size,
                                          sampler=self.sampler,
                                          collate_fn=pred_sequence_collate)
        elif self.dataset_kind == 'ptb':
            # Download "Penn Treebank" dataset
            penn_treebank_dataset(self.data_dir + '/PTB', train=True)
            # Encode
            corpus = lang_util.Corpus(self.data_dir + '/PTB')
            self.train_loader = DataLoader(corpus.train,
                                           batch_size=pred_batch_size,
                                           collate_fn=language_pred_sequence_collate)
            self.test_loader = DataLoader(corpus.valid,
                                          batch_size=pred_batch_size,
                                          collate_fn=language_pred_sequence_collate)

    def _get_loss_function(self):
        # self.loss = None
        # if self.loss_function == "Perplexity":
        #     self.loss = lang_util.Perplexity()
        # else:
        self.loss = getattr(torch.nn, self.loss_function)(reduction='mean')
        self.predictor_loss = None
        if self.model.predictor:
            # Currently using same loss function for both predictor and RSM
            self.predictor_loss = torch.nn.CrossEntropyLoss()

    def model_setup(self, config):
        seed = config.get("seed", random.randint(0, 10000))
        if torch.cuda.is_available():
            print("setup: Using cuda")
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
        else:
            print("setup: Using cpu")
            self.device = torch.device("cpu")

        # Build sampler / data loader
        self._build_dataloader()

        # Build model and optimizer
        self.d_in = reduce(lambda x, y: x * y, self.input_size)
        self.model = RSMLayer(d_in=self.d_in,
                              d_out=self.d_in,
                              m=self.m_groups,
                              n=self.n_cells_per_group,
                              k=self.k_winners,
                              k_winner_cells=self.k_winner_cells,
                              cell_winner_softmax=self.cell_winner_softmax,
                              gamma=self.gamma,
                              eps=self.eps,
                              activation_fn=self.activation_fn,
                              active_dendrites=self.active_dendrites,
                              col_output_cells=self.col_output_cells,
                              embed_dim=self.embed_dim,
                              vocab_size=self.vocab_size,
                              predictor_hidden_size=self.predictor_hidden_size,
                              predictor_output_size=self.predictor_output_size,
                              debug=self.debug)

        self.model.to(self.device)

        self._get_loss_function()

        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.learning_rate,
                                             momentum=self.momentum)

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

    def _eval(self):
        ret = {}
        if self.test_loader:
            with torch.no_grad():
                total_loss = 0.0
                for batch_idx, (inputs, targets, target_labels) in enumerate(self.test_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    x_a_next, x_bs, predictor_outs, phi, psi = self.model(inputs)

                    if targets.dim() > 1:
                        targets = targets.reshape(self.batch_size, self.d_in)

                    if self.model.embedding:
                        # Target is embedded input x
                        targets = self.model.embedding(targets).detach()

                    batch_loss = self.loss(x_a_next, targets)
                    total_loss += batch_loss.item()
                    if batch_idx >= self.batches_in_epoch:
                        break
                ret['test_loss'] = total_loss / (batch_idx + 1)
                if self.dataset_kind == 'ptb':
                    ret['test_ppl'] = math.exp(ret['test_loss'])
        return ret

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

        # Prediction performance
        total_samples = 0.0
        correct_samples = 0.0

        for batch_idx, (inputs, targets, target_labels) in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            # Forward
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            target_labels = target_labels.to(self.device)
            x_a_next, x_bs, predictor_outs, phi, psi = self.model(inputs)

            if targets.dim() > 1:
                targets = targets.reshape(self.batch_size, self.d_in)

            if self.model.embedding:
                # Target is embedded input x
                targets = self.model.embedding(targets).detach()

            batch_loss = self.loss(x_a_next, targets)
            total_loss += batch_loss.item()

            # Backward pass + optimize
            batch_loss.backward()
            # self.loss.detach()  # Possibly needed to reduce memory reqs

            # Backward for predictor
            if self.model.predictor:
                pred_batch_loss = self.predictor_loss(predictor_outs, target_labels)
                _, class_predictions = torch.max(predictor_outs, 1)
                total_samples += target_labels.size(0)
                correct_samples += (class_predictions == target_labels).sum().item()
                ret['pred_acc'] = 100 * correct_samples / total_samples
                ret['pred_loss'] = pred_batch_loss.item()
                pred_batch_loss.backward()

            self.optimizer.step()

            # Update board images on batch 10, each 5 epochs
            if epoch % 5 == 0 and batch_idx == 10:
                if self.dataset_kind == 'mnist':
                    ret['img_inputs'] = self._image_grid(inputs).cpu()
                    ret['img_preds'] = self._image_grid(x_a_next).cpu()
                ret['hist_w_a'] = self.model.linear_a.weight.cpu()
                ret['hist_w_b'] = self.model.linear_b.weight.cpu()
                ret['hist_w_d'] = self.model.linear_d.weight.cpu()

            if batch_idx >= self.batches_in_epoch:
                print("Stopping after %d batches in epoch %d" % (self.batches_in_epoch, epoch))
                break

            if self.batch_log_interval and batch_idx % self.batch_log_interval == 0:
                print("Finished batch %d, batch loss: %.5f" % (batch_idx, batch_loss.item()))

        if epoch % 5 == 0:
            # Evaluate each 5 epochs
            ret.update(self._eval())

        train_time = time.time() - t1
        self._post_epoch(epoch)

        ret["stop"] = 0

        ret['train_loss'] = total_loss / (batch_idx + 1)
        ret["epoch_time_train"] = train_time
        ret["epoch_time"] = time.time() - t1
        ret["learning_rate"] = self.learning_rate
        print(epoch, ret)
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
        checkpoint_file = os.path.join(checkpoint_dir, self.model_filename)

        if checkpoint_file.endswith(".pt"):
            torch.save(self.model, checkpoint_file)
        else:
            torch.save(self.model.state_dict(), checkpoint_file)

        if self.save_onnx_graph_at_checkpoint:
            dummy_input = (torch.rand(1, 1, 28, 28),)
            torch.onnx.export(self.model, dummy_input, self.graph_filename, 
                              verbose=True)

        return checkpoint_file

    def model_restore(self, checkpoint_path):
        """
        :param checkpoint_path: Loads model from this checkpoint path.
        If path is a directory, will append the parameter model_filename
        """
        print("Loading from", checkpoint_path)
        checkpoint_file = os.path.join(checkpoint_path, self.model_filename)
        # Use the slow method if filename ends with .pt
        if checkpoint_file.endswith(".pt"):
            self.model = torch.load(checkpoint_file, map_location=self.device)
        else:
            self.model.load_state_dict(
                torch.load(checkpoint_file, map_location=self.device)
            )
        return self.model

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
