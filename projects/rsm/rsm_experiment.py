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
from ptb_lstm import LSTMModel
from rsm import RSMLayer, RSMPredictor
from rsm_samplers import (
    MNISTSequenceSampler, 
    pred_sequence_collate, 
    LangSequenceSampler,
    language_pred_sequence_collate
)

torch.autograd.set_detect_anomaly(True)


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
        self.pred_model_filename = config.get("pred_model_filename", "pred_model.pth")
        self.graph_filename = config.get("graph_filename", "rsm.onnx")
        self.save_onnx_graph_at_checkpoint = config.get("save_onnx_graph_at_checkpoint", False)
        self.exp_name = config.get("name", "exp")
        self.batch_log_interval = config.get("batch_log_interval", 0)
        self.eval_interval = config.get("eval_interval", 5)
        self.model_kind = config.get("model_kind", "rsm")
        self.debug = config.get("debug", False)
        self.writer = None

        self.iterations = config.get("iterations", 200)
        self.dataset_kind = config.get("dataset", "mnist")

        # Training / testing parameters
        self.batch_size = config.get("batch_size", 128)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.eval_batches_in_epoch = config.get("eval_batches_in_epoch", self.batches_in_epoch)
        self.seq_length = config.get("seq_length", 35)

        # Data parameters
        self.input_size = config.get("input_size", (1, 28, 28))
        self.sequences = config.get("sequences", [[0, 1, 2, 3]])
        self.randomize_sequences = config.get("randomize_sequences", False)

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

        # Training state
        self.best_val_loss = None
        self.do_anneal_learning = False

    def _build_dataloader(self):
        # Extra element for sequential prediction labels

        self.val_loader = None
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

            train_sampler = MNISTSequenceSampler(self.dataset, 
                                                 sequences=self.sequences,
                                                 batch_size=self.batch_size,
                                                 randomize_sequences=self.randomize_sequences)
            self.train_loader = DataLoader(self.dataset,
                                           batch_sampler=train_sampler,
                                           collate_fn=pred_sequence_collate)
            val_sampler = MNISTSequenceSampler(self.dataset, 
                                               sequences=self.sequences,
                                               batch_size=self.batch_size,
                                               randomize_sequences=self.randomize_sequences)
            self.val_loader = DataLoader(self.test_dataset,
                                         batch_sampler=val_sampler,
                                         collate_fn=pred_sequence_collate)
        elif self.dataset_kind == 'ptb':
            # Download "Penn Treebank" dataset
            penn_treebank_dataset(self.data_dir + '/PTB', train=True)
            # Encode
            corpus = lang_util.Corpus(self.data_dir + '/PTB')
            train_sampler = LangSequenceSampler(corpus.train, batch_size=self.batch_size, 
                                                seq_length=self.seq_length)
            self.train_loader = DataLoader(corpus.train,
                                           batch_sampler=train_sampler,
                                           collate_fn=language_pred_sequence_collate)
            val_sampler = LangSequenceSampler(corpus.test, batch_size=self.batch_size,
                                              seq_length=self.seq_length)
            self.val_loader = DataLoader(corpus.test,
                                         batch_sampler=val_sampler,
                                         collate_fn=language_pred_sequence_collate)

    def _get_loss_function(self):
        self.loss = getattr(torch.nn, self.loss_function)(reduction='mean')
        self.predictor_loss = None
        if self.predictor:
            self.predictor_loss = torch.nn.CrossEntropyLoss()

    def _get_optimizer(self):
        self.pred_optimizer = None
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate)
            if self.predictor:
                self.pred_optimizer = torch.optim.Adam(self.predictor.parameters(),
                                                       lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.learning_rate,
                                             momentum=self.momentum)
            if self.predictor:
                self.pred_optimizer = torch.optim.SGD(self.predictor.parameters(),
                                                      lr=self.learning_rate,
                                                      momentum=self.momentum)

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
        self.d_out = config.get("output_size", self.d_in)

        self.predictor = None
        if self.model_kind == "rsm":
            self.model = RSMLayer(d_in=self.d_in,
                                  d_out=self.d_out,
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
                                  debug=self.debug)

            if self.predictor_hidden_size:
                self.predictor = RSMPredictor(d_in=self.m_groups * self.n_cells_per_group,
                                              d_out=self.predictor_output_size,
                                              hidden_size=self.predictor_hidden_size)
                self.predictor.to(self.device)

        elif self.model_kind == "lstm":
            self.model = LSTMModel(vocab_size=self.vocab_size,
                                   emb_size=self.embed_dim,
                                   nhid=200,
                                   dropout=0.5,
                                   nlayers=2)

        self.model.to(self.device)

        self._get_loss_function()
        self._get_optimizer()

    def _image_grid(self, image_batch):
        batch = image_batch.reshape(self.batch_size, 1, 28, 28)
        # make_grid returns 3 channels? -- mean
        grid = vutils.make_grid(batch, normalize=True, padding=5).mean(dim=0)  
        return grid

    def _repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self._repackage_hidden(v) for v in h)

    def _adjust_learning_rate(self, optimizer, epoch):
        if self.do_anneal_learning:
            self.learning_rate *= self.learning_rate_gamma
            self.do_anneal_learning = False
            print("Reducing learning rate by gamma %.2f to: %.5f" % (self.learning_rate_gamma, self.learning_rate))
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _maybe_resize_outputs_targets(self, x_a_next, targets):
        x_a_next = x_a_next.view(-1, self.vocab_size)
        return x_a_next, targets

    def _do_prediction(self, x_bs, pred_targets, total_samples, correct_samples, 
                       total_pred_loss, train=False):
        if self.predictor:
            predictor_outputs = self.predictor(x_bs.detach())
            predictor_outputs = predictor_outputs.view(-1, predictor_outputs.size(2))
            pred_loss = self.predictor_loss(predictor_outputs, pred_targets)
            _, class_predictions = torch.max(predictor_outputs, 1)
            total_samples += pred_targets.size(0)
            correct_samples += (class_predictions == pred_targets).sum().item()
            total_pred_loss += pred_loss.item()
            if train:
                # Predictor backward + optimize
                pred_loss.backward()
                self.pred_optimizer.step()
        return total_samples, correct_samples, total_pred_loss

    def _eval(self):
        ret = {}
        print("Evaluating...")
        self.model.eval()
        if self.predictor:
            self.predictor.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_samples = 0.0
            correct_samples = 0.0
            total_pred_loss = 0.0

            hidden = self.model.init_hidden(self.batch_size)

            for batch_idx, (inputs, targets, pred_targets) in enumerate(self.val_loader):

                # Forward
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pred_targets = pred_targets.to(self.device)
                x_a_next, hidden, x_bs = self.model(inputs, hidden)

                # Loss
                x_a_next, targets = self._maybe_resize_outputs_targets(x_a_next, targets)
                loss = self.loss(x_a_next, targets)
                total_loss += loss.item()

                total_samples, correct_samples, total_pred_loss = self._do_prediction(
                    x_bs, pred_targets, total_samples, correct_samples, total_pred_loss)

                hidden = self._repackage_hidden(hidden)
                if batch_idx >= self.eval_batches_in_epoch:
                    break

            ret['val_loss'] = val_loss = total_loss / (batch_idx + 1)
            ret['val_ppl'] = lang_util.perpl(val_loss)
            if self.predictor:
                test_pred_loss = total_pred_loss / (batch_idx + 1)
                ret['val_pred_ppl'] = lang_util.perpl(test_pred_loss)
                ret['val_pred_acc'] = 100 * correct_samples / total_samples

            if not self.best_val_loss or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            else:
                if self.learning_rate_gamma:
                    self.do_anneal_learning = True  # Reduce LR during post_epoch

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

        self.model.train()
        if self.predictor:
            self.predictor.train()

        # Performance metrics
        total_loss = total_samples = correct_samples = total_pred_loss = 0.0

        hidden = self.model.init_hidden(self.batch_size)

        for batch_idx, (inputs, targets, pred_targets) in enumerate(self.train_loader):
            # Parallelized inputs are of shape (seq_len, batch, input_size)

            hidden = self._repackage_hidden(hidden)

            self.optimizer.zero_grad()
            if self.pred_optimizer:
                self.pred_optimizer.zero_grad()

            # Forward
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred_targets = pred_targets.to(self.device)
            x_a_next, hidden, x_bs = self.model(inputs, hidden)

            # Loss
            x_a_next, targets = self._maybe_resize_outputs_targets(x_a_next, targets)
            loss = self.loss(x_a_next, targets)
            total_loss += loss.item()

            # RSM backward + optimize
            loss.backward()

            if self.model_kind == "lstm":
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                for p in self.model.parameters():
                    p.data.add_(-self.learning_rate, p.grad.data)
            else:
                self.optimizer.step()

            total_samples, correct_samples, total_pred_loss = self._do_prediction(
                x_bs, pred_targets, total_samples, correct_samples, total_pred_loss,
                train=True)

            # Update board images on batch 10, each 5 epochs
            if epoch % 5 == 0 and batch_idx == 10:
                if self.dataset_kind == 'mnist':
                    ret['img_inputs'] = self._image_grid(inputs).cpu()
                    ret['img_preds'] = self._image_grid(x_a_next).cpu()
                ret.update(self.model._track_weights())

            if batch_idx >= self.batches_in_epoch:
                print("Stopping after %d batches in epoch %d" % (self.batches_in_epoch, epoch))
                break

            if self.batch_log_interval and batch_idx % self.batch_log_interval == 0:
                print("Finished batch %d" % batch_idx)
                if self.predictor:
                    acc = 100 * correct_samples / total_samples
                    print("Partial train predictor accuracy: %.3f%%" % (acc))

        if self.eval_interval and epoch % self.eval_interval == 0:
            # Evaluate each x epochs
            ret.update(self._eval())

        train_time = time.time() - t1
        self._post_epoch(epoch)

        ret["stop"] = 0

        ret['train_loss'] = train_loss = total_loss / (batch_idx + 1)
        ret['train_ppl'] = lang_util.perpl(train_loss)
        if self.predictor:
            train_pred_loss = total_pred_loss / (batch_idx + 1)
            ret['train_pred_ppl'] = lang_util.perpl(train_pred_loss)
            ret['train_pred_acc'] = 100 * correct_samples / total_samples

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
        if self.predictor:
            checkpoint_file = os.path.join(checkpoint_dir, self.pred_model_filename)
            if checkpoint_file.endswith(".pt"):
                torch.save(self.predictor, checkpoint_file)
            else:
                torch.save(self.predictor.state_dict(), checkpoint_file)

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
        if checkpoint_file.endswith(".pt"):
            self.model = torch.load(checkpoint_file, map_location=self.device)
        else:
            self.model.load_state_dict(
                torch.load(checkpoint_file, map_location=self.device)
            )
        checkpoint_file = os.path.join(checkpoint_path, self.pred_model_filename)
        if checkpoint_file.endswith(".pt"):
            self.predictor = torch.load(checkpoint_file, map_location=self.device)
        else:
            self.predictor.load_state_dict(
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
