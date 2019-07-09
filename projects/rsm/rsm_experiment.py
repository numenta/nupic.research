import random
import os
import time
import sys
from functools import reduce, partial

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ptb import lang_util
from ptb.ptb_lstm import LSTMModel
from rsm import RSMLayer, RSMPredictor
from util import (
    fig2img, 
    plot_confusion_matrix, 
    plot_activity, 
    plot_representation_similarity, 
    AdamW
)
from rsm_samplers import (
    MNISTSequenceLoader,
    PTBSequenceSampler,
    ptb_pred_sequence_collate
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
        self.jit_trace = config.get("jit_trace", None)
        self.pred_model_filename = config.get("pred_model_filename", "pred_model.pth")
        self.graph_filename = config.get("graph_filename", "rsm.onnx")
        self.save_onnx_graph_at_checkpoint = config.get("save_onnx_graph_at_checkpoint", False)
        self.exp_name = config.get("name", "exp")
        self.batch_log_interval = config.get("batch_log_interval", 0)
        self.eval_interval = config.get("eval_interval", 5)
        self.model_kind = config.get("model_kind", "rsm")
        self.debug = config.get("debug", False)
        self.plot_gradients = config.get("plot_gradients", False)
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
        self.dropout_p = config.get("dropout_p", 0.5)

        # Tweaks
        self.cell_winner_softmax = config.get("cell_winner_softmax", False)
        self.activation_fn = config.get("activation_fn", 'tanh')
        self.active_dendrites = config.get("active_dendrites", None)
        self.col_output_cells = config.get("col_output_cells", None)
        self.clear_memory_each_subseq = config.get("clear_memory_each_subseq", False)
        self.static_digit = config.get("static_digit", False)
        self.use_mnist_pct = config.get("use_mnist_pct", 1.0)
        self.pred_l2_reg = config.get("pred_l2_reg", 0)
        self.decode_from_full_memory = config.get("decode_from_full_memory", False)
        self.boost_strat = config.get("boost_strat", "rsm_inhibition")
        self.pred_gain = config.get("pred_gain", 1.0)
        self.x_b_norm = config.get("x_b_norm", False)
        self.predict_memory = config.get("predict_memory", False)

        # Predictor network
        self.predictor_hidden_size = config.get("predictor_hidden_size", None)
        self.predictor_output_size = config.get("predictor_output_size", 10)

        # Embeddings for language modeling
        self.embed_dim = config.get("embed_dim", 0)
        self.vocab_size = config.get("vocab_size", 0)

        self.loss_function = config.get("loss_function", "MSELoss")
        self.lr_step_schedule = config.get("lr_step_schedule", None)
        self.learning_rate_gamma = config.get("learning_rate_gamma", 0.1)
        self.learning_rate_min = config.get("learning_rate_min", 0.0)

        # Training state
        self.best_val_loss = None
        self.do_anneal_learning = False

        # Additional state for vis, etc
        self.activity_by_inputs = {}  # 'digit-digit' -> list of distribution arrays

        # Convenience
        self.total_cells = self.m_groups * self.n_cells_per_group

    def _build_dataloader(self):
        # Extra element for sequential prediction labels

        self.val_loader = None
        if self.dataset_kind == 'mnist':
            self.dataset = datasets.MNIST(self.data_dir, download=True,
                                          train=True, transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]),)
            self.val_dataset = datasets.MNIST(self.data_dir, download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ]),)
            self.train_loader = MNISTSequenceLoader(self.dataset, 
                                                    sequences=self.sequences,
                                                    batch_size=self.batch_size,
                                                    randomize_sequences=self.randomize_sequences,
                                                    random_mnist_images=not self.static_digit,
                                                    use_mnist_pct=self.use_mnist_pct)

            if self.static_digit:
                # For static digit paradigm, val & train samplers much match to ensure same digit
                # prototype used for each sequence item.
                self.val_loader = self.train_loader
            else:
                self.val_loader = MNISTSequenceLoader(self.val_dataset, 
                                                      sequences=self.sequences,
                                                      batch_size=self.batch_size,
                                                      randomize_sequences=self.randomize_sequences,
                                                      random_mnist_images=not self.static_digit,
                                                      use_mnist_pct=self.use_mnist_pct)
        elif self.dataset_kind == 'ptb':
            # Download "Penn Treebank" dataset
            from torchnlp.datasets import penn_treebank_dataset
            penn_treebank_dataset(self.data_dir + '/PTB', train=True)
            corpus = lang_util.Corpus(self.data_dir + '/PTB')
            train_sampler = PTBSequenceSampler(corpus.train, batch_size=self.batch_size,
                                               seq_length=self.seq_length)

            embedding = lang_util.BitwiseWordEmbedding()

            collate_fn = partial(ptb_pred_sequence_collate, vector_dict=embedding.embedding_dict)
            self.train_loader = DataLoader(corpus.train,
                                           batch_sampler=train_sampler,
                                           collate_fn=collate_fn)
            val_sampler = PTBSequenceSampler(corpus.test, batch_size=self.batch_size,
                                             seq_length=self.seq_length)
            self.val_loader = DataLoader(corpus.test,
                                         batch_sampler=val_sampler,
                                         collate_fn=collate_fn)

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
                                                       lr=self.learning_rate,
                                                       weight_decay=self.pred_l2_reg)
        elif self.optimizer_type == 'adamw':
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=self.learning_rate)
            if self.predictor:
                self.pred_optimizer = AdamW(self.model.parameters(),
                                            lr=self.learning_rate,
                                            weight_decay=self.pred_l2_reg)
 
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.learning_rate,
                                             momentum=self.momentum)
            if self.predictor:
                self.pred_optimizer = torch.optim.SGD(self.predictor.parameters(),
                                                      lr=self.learning_rate,
                                                      momentum=self.momentum,
                                                      weight_decay=self.pred_l2_reg)

    def model_setup(self, config):
        seed = config.get("seed", random.randint(0, 10000))
        if torch.cuda.is_available():
            print("setup: Using cuda")
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
        else:
            print("setup: Using cpu")
            self.device = torch.device("cpu")

        self._build_dataloader()

        # Build model and optimizer
        self.d_in = reduce(lambda x, y: x * y, self.input_size)
        if self.predict_memory:
            self.d_out = self.total_cells
        else:
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
                                  dropout_p=self.dropout_p,
                                  decode_from_full_memory=self.decode_from_full_memory,
                                  x_b_norm=self.x_b_norm,
                                  predict_memory=self.predict_memory,
                                  boost_strat=self.boost_strat,
                                  embed_dim=self.embed_dim,
                                  vocab_size=self.vocab_size,
                                  bsz=self.batch_size,
                                  seq_length=self.seq_length,
                                  debug=self.debug)
            if self.jit_trace:
                # Trace model (Can produce ~25% speed improvement)
                inputs = torch.rand(self.seq_length, self.batch_size, self.d_in)
                hidden = self._init_hidden(self.batch_size)
                print(">> Running JIT trace...")
                self.model = torch.jit.trace(self.model, (inputs, hidden))

            if self.predictor_hidden_size:
                self.predictor = RSMPredictor(d_in=self.m_groups * self.n_cells_per_group,
                                              d_out=self.predictor_output_size,
                                              hidden_size=self.predictor_hidden_size)
                self.predictor.to(self.device)

        elif self.model_kind == "lstm":
            self.model = LSTMModel(vocab_size=self.vocab_size,
                                   embed_dim=self.embed_dim,
                                   nhid=self.m_groups,
                                   d_in=self.d_in,
                                   d_out=self.d_out,
                                   dropout=0.5,
                                   nlayers=2)

        self.model.to(self.device)

        self._get_loss_function()
        self._get_optimizer()

    def _image_grid(self, image_batch, n_seqs=6, compare_with=None, compare_correct=None, limit_seqlen=50):
        '''
        image_batch: n_batches x batch_size x image_dim
        '''
        side = 28
        image_batch = image_batch[:, :n_seqs].reshape(-1, 1, side, side)
        if compare_with is not None:
            # Interleave comparison images with image_batch
            compare_with = compare_with[:, :n_seqs].reshape(-1, 1, side, side)
            max_val = compare_with.max()
            if compare_correct is not None:
                # Add 'incorrect label' to each image (masked by inverse of compare_correct)
                # as 2x2 square 'dot' in upper left corner of falsely predicted targets
                dot_size = 4
                gap = 2
                incorrect = ~compare_correct[:, :n_seqs].transpose(0, 1).flatten()
                compare_with[incorrect, :, gap:gap + dot_size, gap:gap + dot_size] = max_val
            batch = torch.empty((image_batch.shape[0] + compare_with.shape[0], image_batch.shape[1], side, side))
            batch[::2, :, :] = image_batch
            batch[1::2, :, :] = compare_with
        else:
            batch = image_batch
        # make_grid returns 3 channels -- mean since grayscale
        grid = vutils.make_grid(batch[:2*limit_seqlen*n_seqs], normalize=True, nrow=n_seqs * 2, padding=5).mean(dim=0)  
        return grid

    def _plot_grad_flow(self):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    def _repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self._repackage_hidden(v) for v in h)

    def _adjust_learning_rate(self, epoch):
        if self.do_anneal_learning and self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_gamma
            self.do_anneal_learning = False
            print("Reducing learning rate by gamma %.2f to: %.5f" % (self.learning_rate_gamma, self.learning_rate))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _track_weights(self):
        ret = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                data = param.data.cpu()
                ret['hist_' + name] = data
                if self.debug:
                    print("%s: mean: %.3f std: %.3f" % (name, data.mean(), data.std()))
        return ret

    def _init_hidden(self, batch_size):
        if self.model_kind == 'rsm':
            param = next(self.model.parameters())
            x_b = param.new_zeros((batch_size, self.total_cells), dtype=torch.float32, requires_grad=False)
            phi = param.new_zeros((batch_size, self.total_cells), dtype=torch.float32, requires_grad=False)
            psi = param.new_zeros((batch_size, self.total_cells), dtype=torch.float32, requires_grad=False)
            return (x_b, phi, psi)
        elif self.model_kind == 'lstm':
            return self.model.init_hidden(batch_size)

    def _maybe_resize_outputs_targets(self, x_a_next, targets):
        x_a_next = x_a_next.view(-1, self.d_out)
        targets = targets.view(-1, self.d_out)
        return x_a_next, targets

    def _store_activity_for_viz(self, x_bs, input_labels, pred_labels):
        """
        Aggregate activity for a supplied batch
        """
        for _x_b, label, target in zip(x_bs, input_labels, pred_labels):
            _label = label.item()
            _label_next = target.item()
            activity = _x_b.detach().view(self.m_groups, -1)
            key = "%d-%d" % (_label, _label_next)
            if key not in self.activity_by_inputs:
                self.activity_by_inputs[key] = []
            self.activity_by_inputs[key].append(activity)

    def _do_prediction(self, x_b, pred_targets, total_samples, correct_samples, 
                       total_pred_loss, train=False):
        cm_fig = None
        if self.predictor:
            bs, tc = x_b.size()
            pred_targets = pred_targets.flatten()
            predictor_outputs = self.predictor(x_b.detach())  # (bs, total_cells) -> (bs, n_labels)
            pred_loss = self.predictor_loss(predictor_outputs, pred_targets)
            _, class_predictions = torch.max(predictor_outputs, 1)
            total_samples += pred_targets.size(0)
            correct_arr = (class_predictions == pred_targets)
            correct_samples += correct_arr.sum().item()
            total_pred_loss += pred_loss.item()
            if train:
                # Predictor backward + optimize
                pred_loss.backward()
                self.pred_optimizer.step()
        return total_samples, correct_samples, class_predictions, correct_arr, total_pred_loss, cm_fig

    def _confusion_matrix(self, pred_targets, class_predictions):
        class_names = [str(x) for x in range(self.predictor_output_size)]
        cm_ax, cm_fig = plot_confusion_matrix(pred_targets, class_predictions, class_names, title="Prediction Confusion")
        return cm_fig

    def _eval(self):
        ret = {}
        print("Evaluating...")
        # Disable dropout
        self.model.eval()  
        if self.predictor:
            self.predictor.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_samples = 0.0
            correct_samples = 0.0
            total_pred_loss = 0.0

            hidden = self._init_hidden(self.batch_size)

            all_x_a_next = all_targets = all_correct_arrs = all_pred_targets = all_class_predictions = None

            for batch_idx, (inputs, targets, pred_targets, input_labels) in enumerate(self.val_loader):

                # Forward
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pred_targets = pred_targets.to(self.device)
                x_a_next, hidden = self.model(inputs, hidden)

                # Loss
                loss = self.loss(x_a_next, targets)
                total_loss += loss.item()

                x_bs = hidden[0]

                total_samples, correct_samples, class_predictions, correct_arr, total_pred_loss, cm_fig = self._do_prediction(
                    x_bs, pred_targets, total_samples, correct_samples, total_pred_loss)

                hidden = self._repackage_hidden(hidden)

                # Save results for image grid & confusion matrix
                x_a_next.unsqueeze_(0)
                targets.unsqueeze_(0)
                correct_arr.unsqueeze_(0)
                all_x_a_next = x_a_next if all_x_a_next is None else torch.cat((all_x_a_next, x_a_next))
                all_targets = targets if all_targets is None else torch.cat((all_targets, targets))
                all_correct_arrs = correct_arr if all_correct_arrs is None else torch.cat((all_correct_arrs, correct_arr))
                all_pred_targets = pred_targets if all_pred_targets is None else torch.cat((all_pred_targets, pred_targets))                
                all_class_predictions = class_predictions if all_class_predictions is None else torch.cat((all_class_predictions, class_predictions))                

                if self.dataset_kind == 'mnist' and self.model_kind == 'rsm':
                    # Summary of column activation by input & next input
                    self._store_activity_for_viz(x_bs, input_labels, pred_targets)

                ret.update(self._track_weights())

                if batch_idx >= self.eval_batches_in_epoch:
                    break

            if self.dataset_kind == 'mnist' and self.model_kind == 'rsm':
                ret['img_preds'] = self._image_grid(all_x_a_next, 
                                                    compare_with=all_targets, 
                                                    compare_correct=all_correct_arrs).cpu()
                cm_fig = self._confusion_matrix(all_pred_targets, all_class_predictions)
                ret['img_confusion'] = fig2img(cm_fig)
                col_activity_grid = plot_activity(self.activity_by_inputs, 
                                                  n_labels=self.predictor_output_size,
                                                  level='cell')
                img_repr_sim = plot_representation_similarity(self.activity_by_inputs,
                                                              n_labels=self.predictor_output_size,
                                                              title=self.boost_strat)
                ret['img_repr_sim'] = fig2img(img_repr_sim)
                ret['img_col_activity'] = fig2img(col_activity_grid)
                self.activity_by_inputs = {}

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

        self.model.train()  # Needed if using dropout
        if self.predictor:
            self.predictor.train()

        # Performance metrics
        total_loss = total_samples = correct_samples = total_pred_loss = 0.0

        hidden = self._init_hidden(self.batch_size)

        for batch_idx, (inputs, targets, pred_targets, _) in enumerate(self.train_loader):
            # Parallelized inputs are of shape (seq_len, batch, input_size)

            hidden = self._repackage_hidden(hidden)

            self.optimizer.zero_grad()
            if self.pred_optimizer:
                self.pred_optimizer.zero_grad()

            # Forward
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred_targets = pred_targets.to(self.device)

            x_a_next, hidden = self.model(inputs, hidden)

            # Loss
            loss = self.loss(x_a_next, targets)
            total_loss += loss.item()

            if self.debug:
                self.model._register_hooks()

            # RSM backward + optimize
            loss.backward()

            if self.model_kind == "lstm":
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                for p in self.model.parameters():
                    p.data.add_(-self.learning_rate, p.grad.data)
            else:
                self.optimizer.step()

            if self.plot_gradients:
                self._plot_gradient_flow()

            x_b = hidden[0]
            total_samples, correct_samples, class_predictions, correct_arr, total_pred_loss, _ = self._do_prediction(
                x_b, pred_targets, total_samples, correct_samples, total_pred_loss,
                train=True)

            if batch_idx > self.batches_in_epoch:
                print("Stopping after %d batches in epoch %d" % (self.batches_in_epoch, epoch))
                break

            if self.batch_log_interval and batch_idx % self.batch_log_interval == 0:
                print("Finished batch %d" % batch_idx)
                if self.predictor:
                    acc = 100 * correct_samples / total_samples
                    print("Partial train predictor accuracy: %.3f%%" % (acc))

            if self.clear_memory_each_subseq:
                hidden = self._init_hidden(self.batch_size)

        if self.eval_interval and (epoch == 0 or (epoch + 1) % self.eval_interval == 0):
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
        self._adjust_learning_rate(epoch)

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
