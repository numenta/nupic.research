#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/

import os
import random
import sys
import time
from functools import partial, reduce

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms

from ptb import lang_util
from baseline_models import LSTMModel, RNNModel
from nupic.torch.modules.k_winners import KWinners
from rsm import RSMNet, RSMPredictor
from rsm_samplers import (
    MNISTBufferedDataset,
    MNISTSequenceSampler,
    PTBSequenceSampler,
    pred_sequence_collate,
    ptb_pred_sequence_collate,
)
from util import (
    fig2img,
    plot_activity,
    plot_activity_grid,
    plot_confusion_matrix,
    plot_representation_similarity,
    plot_tensors,
    print_aligned_sentences,
    print_epoch_values,
    SmoothedCrossEntropyLoss
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
        self.save_onnx_graph_at_checkpoint = config.get(
            "save_onnx_graph_at_checkpoint", False
        )
        self.exp_name = config.get("name", "exp")
        self.batch_log_interval = config.get("batch_log_interval", 0)
        self.eval_interval = config.get("eval_interval", 5)
        self.eval_interval_schedule = config.get("eval_interval_schedule", None)
        self.model_kind = config.get("model_kind", "rsm")
        self.debug = config.get("debug", False)
        self.visual_debug = config.get("visual_debug", False)

        # Instrumentation
        self.instrumentation = config.get("instrumentation", False)
        self.plot_gradients = config.get("plot_gradients", False)
        self.instr_charts = config.get("instr_charts", [])

        self.iterations = config.get("iterations", 200)
        self.dataset_kind = config.get("dataset", "mnist")

        # Training / testing parameters
        self.batch_size = config.get("batch_size", 128)
        self.eval_batch_size = config.get("eval_batch_size", self.batch_size)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.batches_in_first_epoch = config.get("batches_in_first_epoch", self.batches_in_epoch)
        self.eval_batches_in_epoch = config.get(
            "eval_batches_in_epoch", self.batches_in_epoch
        )
        self.pause_after_upticks = config.get("pause_after_upticks", 0)
        self.pause_after_epochs = config.get("pause_after_epochs", 0)
        self.pause_eval_interval = config.get("pause_eval_interval", 10)
        self.pause_min_epoch = config.get("pause_min_epoch", 0)

        # Data parameters
        self.input_size = config.get("input_size", (1, 28, 28))
        self.sequences = config.get("sequences", [[0, 1, 2, 3]])

        self.learning_rate = config.get("learning_rate", 0.0005)
        self.pred_learning_rate = config.get("pred_learning_rate", self.learning_rate)
        self.momentum = config.get("momentum", 0.9)
        self.optimizer_type = config.get("optimizer", "adam")
        self.pred_optimizer_type = config.get("pred_optimizer", self.optimizer_type)

        # Model
        self.m_groups = config.get("m_groups", 200)
        self.n_cells_per_group = config.get("n_cells_per_group", 6)
        self.k_winners = config.get("k_winners", 25)
        self.k_winners_pct = config.get("k_winners_pct", None)
        if self.k_winners_pct is not None:
            # Optionally define k-winners proportionally
            self.k_winners = int(self.m_groups * self.k_winners_pct)
        self.gamma = config.get("gamma", 0.5)
        self.eps = config.get("eps", 0.5)
        self.k_winner_cells = config.get("k_winner_cells", 1)
        self.flattened = self.n_cells_per_group == 1
        self.forget_mu = config.get("forget_mu", 0.0)

        # Tweaks
        self.activation_fn = config.get("activation_fn", "tanh")
        self.decode_activation_fn = config.get("decode_activation_fn", None)
        self.static_digit = config.get("static_digit", False)
        self.randomize_sequence_cursors = config.get("randomize_sequence_cursors", True)
        self.use_mnist_pct = config.get("use_mnist_pct", 1.0)
        self.pred_l2_reg = config.get("pred_l2_reg", 0)
        self.l2_reg = config.get("l2_reg", 0)
        self.dec_l2_reg = config.get("dec_l2_reg", 0)
        self.decode_from_full_memory = config.get("decode_from_full_memory", False)
        self.boost_strat = config.get("boost_strat", "rsm_inhibition")
        self.x_b_norm = config.get("x_b_norm", False)
        self.mask_shifted_pi = config.get("mask_shifted_pi", False)
        self.boost_strength = config.get("boost_strength", 1.0)
        self.boost_strength_factor = config.get("boost_strength_factor", 1.0)
        self.duty_cycle_period = config.get("duty_cycle_period", 1000)
        self.mult_integration = config.get("mult_integration", False)
        self.noise_buffer = config.get("noise_buffer", False)
        self.col_output_cells = config.get("col_output_cells", False)
        self.fpartition = config.get("fpartition", None)
        self.balance_part_winners = config.get("balance_part_winners", False)
        self.weight_sparsity = config.get("weight_sparsity", None)
        self.embedding_kind = config.get("embedding_kind", "rsm_bitwise")
        self.feedback_conn = config.get("feedback_conn", False)
        self.input_bias = config.get("input_bias", False)
        self.decode_bias = config.get("decode_bias", True)
        self.loss_layers = config.get("loss_layers", "first")
        self.top_lateral_conn = config.get("top_lateral_conn", True)
        self.lateral_conn = config.get("lateral_conn", True)
        self.trainable_decay = config.get("trainable_decay", False)
        self.trainable_decay_rec = config.get("trainable_decay_rec", False)
        self.max_decay = config.get("max_decay", 1.0)
        self.additive_decay = config.get("additive_decay", False)
        self.stoch_decay = config.get("stoch_decay", False)
        self.predict_from_input = config.get("predict_from_input", False)
        self.stoch_k_sd = config.get("stoch_k_sd", False)
        self.rec_active_dendrites = config.get("rec_active_dendrites", 0)
        self.mem_floor = config.get("mem_floor", 0.0)

        # Prediction smoothing
        self.word_cache_decay = config.get("word_cache_decay", 0.0)
        self.word_cache_pct = config.get("word_cache_pct", 0.0)
        self.unif_smoothing = config.get("unif_smoothing", 0.0)

        # Predictor network
        self.predictor_hidden_size = config.get("predictor_hidden_size", None)
        self.predictor_output_size = config.get("predictor_output_size", 10)

        self.n_layers = config.get("n_layers", 1)

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
        self.model_learning_paused = False
        self.n_upticks = 0

        self.train_hidden_buffer = []
        self.train_output_buffer = []

        # Additional state for vis, etc
        self.activity_by_inputs = {}  # 'digit-digit' -> list of distribution arrays

    def _build_dataloader(self):
        # Extra element for sequential prediction labels

        self.val_loader = self.corpus = None
        if self.dataset_kind == "mnist":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            self.dataset = MNISTBufferedDataset(
                self.data_dir, download=True, train=True, transform=transform
            )
            self.val_dataset = MNISTBufferedDataset(
                self.data_dir, download=True, transform=transform
            )

            self.train_sampler = MNISTSequenceSampler(
                self.dataset,
                sequences=self.sequences,
                batch_size=self.batch_size,
                random_mnist_images=not self.static_digit,
                randomize_sequence_cursors=self.randomize_sequence_cursors,
                noise_buffer=self.noise_buffer,
                use_mnist_pct=self.use_mnist_pct,
                max_batches=self.batches_in_epoch,
            )

            if self.static_digit:
                # For static digit paradigm, val & train samplers much
                # match to ensure same digit prototype used for each sequence item.
                self.val_sampler = self.train_sampler
            else:
                self.val_sampler = MNISTSequenceSampler(
                    self.val_dataset,
                    sequences=self.sequences,
                    batch_size=self.batch_size,
                    random_mnist_images=not self.static_digit,
                    randomize_sequence_cursors=self.randomize_sequence_cursors,
                    noise_buffer=self.noise_buffer,
                    use_mnist_pct=self.use_mnist_pct,
                    max_batches=self.eval_batches_in_epoch,
                )
            self.train_loader = DataLoader(
                self.dataset,
                batch_sampler=self.train_sampler,
                collate_fn=pred_sequence_collate,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_sampler=self.val_sampler,
                collate_fn=pred_sequence_collate,
            )

        elif self.dataset_kind == "ptb":
            # Download "Penn Treebank" dataset
            from torchnlp.datasets import penn_treebank_dataset

            print("Maybe download PTB...")
            penn_treebank_dataset(self.data_dir + "/PTB", train=True, test=True)
            corpus = lang_util.Corpus(self.data_dir + "/PTB")
            train_sampler = PTBSequenceSampler(
                corpus.train,
                batch_size=self.batch_size,
                max_batches=self.batches_in_epoch,
            )

            if self.embedding_kind == "rsm_bitwise":
                embedding = lang_util.BitwiseWordEmbedding().embedding_dict
            elif self.embedding_kind in ["bpe", "glove"]:
                from torchnlp.word_to_vector import BPEmb, GloVe
                cache_dir = self.data_dir + "/torchnlp/.word_vectors_cache"
                if self.embedding_kind == "bpe":
                    vectors = BPEmb(dim=self.embed_dim, cache=cache_dir)
                else:
                    vectors = GloVe(name="6B", dim=self.embed_dim, cache=cache_dir)
                embedding = {}
                for word_id, word in enumerate(corpus.dictionary.idx2word):
                    embedding[word_id] = vectors[word]
            elif "ptb_fasttext" in self.embedding_kind:
                import fasttext
                # Generated via notebooks/ptb_embeddings.ipynb
                embedding = {}
                ft_model = fasttext.load_model(self.data_dir + "/embeddings/%s.bin" % self.embedding_kind)
                for word_id, word in enumerate(corpus.dictionary.idx2word):
                    embedding[word_id] = torch.tensor(ft_model[word])

            collate_fn = partial(ptb_pred_sequence_collate, vector_dict=embedding)
            self.train_loader = DataLoader(
                corpus.train, batch_sampler=train_sampler, collate_fn=collate_fn
            )
            val_sampler = PTBSequenceSampler(
                corpus.test,
                batch_size=self.eval_batch_size,
                max_batches=self.eval_batches_in_epoch,
                uniform_offsets=True
            )
            self.val_loader = DataLoader(
                corpus.test, batch_sampler=val_sampler, collate_fn=collate_fn
            )
            self.corpus = corpus
            print("Built dataloaders...")

    def _get_loss_function(self):
        self.loss = getattr(torch.nn, self.loss_function)(reduction="mean")
        self.predictor_loss = None
        if self.predictor:
            self.predictor_loss = torch.nn.NLLLoss()

    def _get_one_optimizer(self, type, params, lr, l2_reg=0.0):
        if type == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=l2_reg,
            )
        elif type == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=lr, 
                momentum=self.momentum,
                weight_decay=l2_reg
            )
        return optimizer

    def _get_optimizer(self):
        self.pred_optimizer = None
        self.optimizer = self._get_one_optimizer(self.optimizer_type, 
                                                 self.model.parameters(), 
                                                 self.learning_rate)
        if self.predictor:
            self.pred_optimizer = self._get_one_optimizer(self.pred_optimizer_type, 
                                                     self.predictor.parameters(), 
                                                     self.pred_learning_rate, l2_reg=self.pred_l2_reg)

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
        self.d_out = config.get("output_size", self.d_in)
        self.predictor = None
        predictor_d_in = self.m_groups
        self.model = RSMNet(
            n_layers=self.n_layers,
            d_in=self.d_in,
            d_out=self.d_out,
            m=self.m_groups,
            n=self.n_cells_per_group,
            k=self.k_winners,
            k_winner_cells=self.k_winner_cells,
            gamma=self.gamma,
            eps=self.eps,
            forget_mu=self.forget_mu,
            activation_fn=self.activation_fn,
            decode_activation_fn=self.decode_activation_fn,
            decode_from_full_memory=self.decode_from_full_memory,
            col_output_cells=self.col_output_cells,
            x_b_norm=self.x_b_norm,
            mask_shifted_pi=self.mask_shifted_pi,
            boost_strat=self.boost_strat,
            boost_strength=self.boost_strength,
            boost_strength_factor=self.boost_strength_factor,
            duty_cycle_period=self.duty_cycle_period,
            weight_sparsity=self.weight_sparsity,
            mult_integration=self.mult_integration,
            fpartition=self.fpartition,
            balance_part_winners=self.balance_part_winners,
            feedback_conn=self.feedback_conn,
            lateral_conn=self.lateral_conn,
            top_lateral_conn=self.top_lateral_conn,
            input_bias=self.input_bias,
            decode_bias=self.decode_bias,
            trainable_decay=self.trainable_decay,
            trainable_decay_rec=self.trainable_decay_rec,
            max_decay=self.max_decay,
            additive_decay=self.additive_decay,
            stoch_decay=self.stoch_decay,
            embed_dim=self.embed_dim,
            vocab_size=self.vocab_size,
            stoch_k_sd=self.stoch_k_sd,
            rec_active_dendrites=self.rec_active_dendrites,
            mem_floor=self.mem_floor,
            debug=self.debug,
            visual_debug=self.visual_debug
        )
        if self.predict_from_input:
            predictor_d_in = self.d_in
        else:
            if self.n_layers > 1:
                predictor_d_in = sum([l.total_cells for l in self.model.children()])
            else:
                predictor_d_in = self.m_groups * self.n_cells_per_group

        self.model.to(self.device)

        if self.predictor_hidden_size:
            self.predictor = RSMPredictor(
                d_in=predictor_d_in,
                d_out=self.predictor_output_size,
                hidden_size=self.predictor_hidden_size
            )
            self.predictor.to(self.device)

        self._get_loss_function()
        self._get_optimizer()

        if self.word_cache_decay:
            self.word_cache = torch.zeros((self.eval_batch_size, self.vocab_size), device=self.device)        

    def _image_grid(
        self,
        image_batch,
        n_seqs=6,
        max_seqlen=50,
        compare_with=None,
        compare_correct=None,
        limit_seqlen=50,
        side=28
    ):
        """
        image_batch: n_batches x batch_size x image_dim
        """
        image_batch = image_batch[:max_seqlen, :n_seqs].reshape(-1, 1, side, side)
        if compare_with is not None:
            # Interleave comparison images with image_batch
            compare_with = compare_with[:max_seqlen, :n_seqs].reshape(-1, 1, side, side)
            max_val = compare_with.max()
            if compare_correct is not None:
                # Add 'incorrect label' to each image (masked by inverse of
                # compare_correct) as 2x2 square 'dot' in upper left corner of falsely
                # predicted targets
                dsize = 4
                gap = 2
                incorrect = ~compare_correct[:max_seqlen, :n_seqs].flatten()
                compare_with[
                    incorrect, :, gap : gap + dsize, gap : gap + dsize
                ] = max_val
            batch = torch.empty(
                (
                    image_batch.shape[0] + compare_with.shape[0],
                    image_batch.shape[1],
                    side,
                    side,
                )
            )
            batch[::2, :, :] = image_batch
            batch[1::2, :, :] = compare_with
        else:
            batch = image_batch
        # make_grid returns 3 channels -- mean since grayscale
        grid = vutils.make_grid(
            batch[: 2 * limit_seqlen * n_seqs],
            normalize=True,
            nrow=n_seqs * 2,
            padding=5,
        ).mean(dim=0)
        return grid

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
            print(
                "Reducing learning rate by gamma %.2f to: %.5f"
                % (self.learning_rate_gamma, self.learning_rate)
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _store_instr_hists(self):
        ret = {}
        if self.instrumentation:
            for name, param in self.model.named_parameters():
                if "weight" in name or "decay" in name:
                    data = param.data.cpu()
                    if data.size(0):
                        ret["hist_" + name] = data
                        if self.debug:
                            print(
                                "%s: mean: %.3f std: %.3f" % (name, data.mean(), data.std())
                            )
        return ret

    def _init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size)

    def _store_activity_for_viz(self, x_bs, input_labels, pred_labels):
        """
        Aggregate activity for a supplied batch
        """
        for _x_b, label, target in zip(x_bs, input_labels, pred_labels):
            _label = label.item()
            _label_next = target.item()
            activity = _x_b.detach().view(self.m_groups, -1).squeeze()
            key = "%d-%d" % (_label, _label_next)
            if key not in self.activity_by_inputs:
                self.activity_by_inputs[key] = []
            self.activity_by_inputs[key].append(activity)

    def _cache_inputs(self, input_labels):
        """
        Word cache for smoothing, currently only used for eval (on test)
        """
        self.word_cache.scatter_(1, input_labels.unsqueeze(1), 1.0)
        # Decay
        self.word_cache = self.word_cache * self.word_cache_decay

    def _get_prediction_and_loss_inputs(self, hidden, output, inputs=None):
        x_b = None
        if self.model_kind == 'rsm':
            # hidden is (x_b, phi, psi, hebb)
            # higher layers predict decaying version of lower layer x_b
            x_b = hidden[0]
            # TODO: Option to train separate predictors on each layer and interpolate
            if self.predictor:
                if self.predict_from_input:
                    predictor_input = inputs
                else:
                    # Predict from concat of all layer hidden states
                    predictor_input = torch.cat(x_b, dim=1).view(-1, self.predictor.d_in).detach()
        elif self.model_kind == 'rnn':
            # hidden is [n_layers x bsz x nhid]
            x_b = hidden
            # For RNN/LSTM, predict from model output
            predictor_input = output.detach()
        elif self.model_kind == 'lstm':
            # hidden is (h [n_layers x bsz x nhid], c [n_layers x bsz x nhid])
            x_b = hidden[0]  # Get hidden state h
            # For RNN/LSTM, predict from model output
            predictor_input = output.detach()
        return x_b, predictor_input

    def _backward_and_optimize(self, loss):
        if self.debug:
            self.model._register_hooks()
        loss.backward()
        if self.model_kind == "lstm":
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            for p in self.model.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)
        else:
            self.optimizer.step()

    def _do_prediction(
        self,
        input,
        pred_targets,
        pcounts,
        train=False,
        batch_idx=0
    ):
        """
        Do prediction. If multiple layers decode from all cells in deepest
        layer.
        """
        class_predictions = correct_arr = None
        if self.predictor:
            pred_targets = pred_targets.flatten()

            predictor_outputs = self.predictor(input.detach())
            predictions = torch.zeros_like(predictor_outputs)

            predictor_mass_pct = 1.0
            if self.word_cache_decay and not train and predictions.size(0) == self.word_cache.size(0):
                # Word cache enabled (for eval)
                mass_pct = self.word_cache_pct
                predictor_mass_pct -= mass_pct
                predictions += mass_pct * self.word_cache / self.word_cache.sum(dim=1, keepdim=True)

            if self.unif_smoothing:
                # Uniform smoothing enabled
                mass_pct = self.unif_smoothing
                predictor_mass_pct -= mass_pct
                predictions += mass_pct * torch.ones_like(predictor_outputs) / self.vocab_size

            predictions += predictor_mass_pct * predictor_outputs

            prediction_log_probs = predictions.log()

            pred_loss = self.predictor_loss(prediction_log_probs, pred_targets)  # NLLLoss
            _, class_predictions = torch.max(predictor_outputs, 1)
            pcounts['total_samples'] += pred_targets.size(0)
            correct_arr = class_predictions == pred_targets
            pcounts['correct_samples'] += correct_arr.sum().item()
            batch_loss = pred_loss.item()
            pcounts['total_pred_loss'] += batch_loss
            if train:
                # Predictor backward + optimize
                pred_loss.backward()
                self.pred_optimizer.step()

        if self.batch_log_interval and batch_idx % self.batch_log_interval == 0:
            print("Finished batch %d" % batch_idx)
            if self.predictor:
                acc = 100 * pcounts['correct_samples'] / pcounts['total_samples']
                batch_acc = correct_arr.float().mean() * 100
                batch_ppl = lang_util.perpl(batch_loss)
                print(
                    "Partial train pred acc - epoch: %.3f%%, "
                    "batch acc: %.3f%%, batch ppl: %.1f"
                    % (acc, batch_acc, batch_ppl)
                )

        return (
            pcounts,
            class_predictions,
            correct_arr
        )

    def _compute_loss(self, predicted_outputs, targets):
        """
        Compute loss across multiple layers (if applicable).

        First layer loss (l1_loss) is between last image prediction and actual input image
        Layers > 1 loss (ls_loss) is between last output (hidden predictions) and actual hidden

        Args:
            - predicted_outputs: list of len n_layers of (bsz, d_in or total_cells)
            - targets: 2-tuple
                - list of actual_input (bsz, d_in) by layer
                - list of x_b (bsz, total_cells)) by layer
            - x_b_target: (n_layers, bsz, total_cells)

        Note that batch size will differ if using a smaller first epoch batch size. 
        In this case we crop target tensors to match predictions.

        TODO: Decision to be made on whether to compute loss vs max-pooled column
        activations or cells.
        """
        loss = None
        if predicted_outputs is not None:

            # TODO: We can stack these and run loss once only
            if self.loss_layers in ['first', 'all_layers']:
                bottom_targets = targets[0].detach()
                pred_img = predicted_outputs[0]
                l1_loss = self.loss(pred_img, bottom_targets)
                if loss is None:
                    loss = l1_loss
                else:
                    loss += l1_loss

            if self.n_layers > 1 and self.loss_layers in ['above_first', 'all_layers']:
                memory = self._repackage_hidden(targets[1])
                for l in range(self.n_layers - 1):
                    higher_targets = memory[l]  # Target memory states up to 2nd to last layer
                    outputs = predicted_outputs[l+1]  # Predictions from layer above
                    ls_loss = self.loss(outputs, higher_targets)
                    if loss is None:
                        loss = ls_loss
                    else:
                        loss += ls_loss

            if self.l2_reg and self.lateral_conn:
                for l in self.model.children():
                    # Add L2 reg term for recurrent weights
                    loss += self.l2_reg * l.linear_b.weight.norm(2) ** 2
                    if hasattr(l.linear_b, 'bias'):
                        loss += self.l2_reg * l.linear_b.bias.norm(2) ** 2
            if self.dec_l2_reg:
                for l in self.model.children():
                    # Add L2 reg term for decode weights
                    loss += self.dec_l2_reg * l.linear_d.weight.norm(2) ** 2
                    if hasattr(l.linear_d, 'bias') and l.linear_d.bias is not None:
                        loss += self.dec_l2_reg * l.linear_d.bias.norm(2) ** 2

        return loss

    def _agg_batch_metrics(self, metrics, **kwargs):
        for metric_key, val in kwargs.items():
            if val is not None:
                if metric_key not in metrics:
                    metrics[metric_key] = val
                else:
                    current = metrics[metric_key]
                    metrics[metric_key] = torch.cat((current, val))
        return metrics

    def _generate_instr_charts(self, metrics):
        ret = {}
        if self.model_kind == "rsm" and self.instrumentation:
            if self.dataset_kind == "mnist":
                if "img_confusion" in self.instr_charts:
                    class_names = [str(x) for x in range(self.predictor_output_size)]
                    cm_ax, cm_fig = plot_confusion_matrix(
                        metrics['pred_targets'], metrics['class_predictions'], class_names, title="Prediction Confusion"
                    )
                    ret["img_confusion"] = fig2img(cm_fig)
                if "img_repr_sim" in self.instr_charts:
                    img_repr_sim = plot_representation_similarity(
                        self.activity_by_inputs,
                        n_labels=self.predictor_output_size,
                        title=self.boost_strat,
                    )
                    ret["img_repr_sim"] = fig2img(img_repr_sim)
                if "img_col_activity" in self.instr_charts:
                    if self.flattened:
                        activity_grid = plot_activity_grid(
                            self.activity_by_inputs, n_labels=self.predictor_output_size
                        )
                    else:
                        activity_grid = plot_activity(
                            self.activity_by_inputs,
                            n_labels=self.predictor_output_size,
                            level="cell",
                        )
                    ret["img_col_activity"] = fig2img(activity_grid)
                self.activity_by_inputs = {}

            if "img_preds" in self.instr_charts:
                ret["img_preds"] = self._image_grid(
                    metrics['pred_images'],
                    compare_with=metrics['targets'],
                    compare_correct=metrics['correct_arr'],
                ).cpu()

            if "img_memory_snapshot" in self.instr_charts and self.n_layers > 1:
                last_inp_layers = [None for x in range(self.n_layers)]
                last_inp_layers[0] = metrics['last_input_snp']
                fig = plot_tensors(self.model, [
                    ('last_out', metrics['last_output_snp']),
                    ('inputs', last_inp_layers),
                    ('x_b', metrics['last_hidden_snp'])
                ], return_fig=True)
                ret["img_memory_snapshot"] = fig2img(fig)

        return ret

    def _read_out_predictions(
        self,
        pred_targets,
        class_predictions,
        read_out_tgt,
        read_out_pred,
        read_out_len=20,
    ):
        if self.predictor and self.corpus and len(read_out_tgt) < read_out_len:
            read_out_tgt.append(pred_targets[0])
            read_out_pred.append(class_predictions[0])
            if len(read_out_tgt) == read_out_len:
                print_aligned_sentences(
                    self.corpus.read_out(read_out_tgt),
                    self.corpus.read_out(read_out_pred),
                    labels=["Targ", "Pred"],
                )

    def eval_epoch(self, epoch):
        ret = {}
        print("Evaluating...")
        # Disable dropout
        self.model.eval()
        if self.predictor:
            self.predictor.eval()

        if self.weight_sparsity is not None:
            # Rezeroing happens before forward pass, so rezero after last
            # training forward.
            self.model._zero_sparse_weights()

        with torch.no_grad():
            total_loss = 0.0
            pcounts = {
                'total_samples': 0.0,
                'correct_samples': 0.0,
                'total_pred_loss': 0.0
            }

            hidden = self._init_hidden(self.eval_batch_size)

            last_output = None
            read_out_tgt = []
            read_out_pred = []
            metrics = {}

            for _b_idx, (inputs, targets, pred_targets, input_labels) in enumerate(
                self.val_loader
            ):

                # Forward
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pred_targets = pred_targets.to(self.device)
                input_labels = input_labels.to(self.device)

                self._cache_inputs(input_labels)

                x_a_next, hidden = self.model(inputs, hidden)

                x_b, pred_input = self._get_prediction_and_loss_inputs(hidden, x_a_next, inputs=inputs)

                # Loss
                loss = self._compute_loss(last_output, (inputs, x_b))
                if loss is not None:
                    total_loss += loss.item()

                pcounts, class_predictions, correct_arr = (
                    self._do_prediction(
                        pred_input, pred_targets, pcounts,
                        batch_idx=_b_idx
                    )
                )

                self._read_out_predictions(
                    pred_targets, class_predictions, read_out_tgt, read_out_pred
                )

                hidden = self._repackage_hidden(hidden)

                if self.instrumentation:
                    metrics = self._agg_batch_metrics(metrics,
                                                      pred_images=x_a_next[0].unsqueeze(0),
                                                      targets=targets.unsqueeze(0),
                                                      correct_arr=correct_arr.unsqueeze(0),
                                                      pred_targets=pred_targets,
                                                      class_predictions=class_predictions)

                    if self.dataset_kind == "mnist" and self.model_kind == "rsm":
                        # Summary of column activation by input & next input
                        self._store_activity_for_viz(x_b, input_labels, pred_targets)

                last_output = x_a_next

            if self.instrumentation:
                x_b_delta = None
                # Save some snapshots from last batch of epoch
                if self.model_kind == "rsm":
                    metrics['last_hidden_snp'] = x_b
                    metrics['last_input_snp'] = inputs
                    metrics['last_output_snp'] = last_output

                # After all eval batches, generate stats & figures
                ret.update(self._generate_instr_charts(metrics))
                ret.update(self._store_instr_hists())

            ret["val_loss"] = val_loss = total_loss / (_b_idx + 1)
            if self.predictor:
                test_pred_loss = pcounts['total_pred_loss'] / (_b_idx + 1)
                ret["val_pred_ppl"] = lang_util.perpl(test_pred_loss)
                ret["val_pred_acc"] = 100 * pcounts['correct_samples'] / pcounts['total_samples']

            if not self.best_val_loss or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            else:
                # Val loss increased
                if self.learning_rate_gamma:
                    self.do_anneal_learning = True  # Reduce LR during post_epoch
                if self.pause_after_upticks and not self.model_learning_paused:
                    if not self.pause_min_epoch or (self.pause_min_epoch and epoch >= self.pause_min_epoch):
                        self.n_upticks += 1
                        if self.n_upticks >= self.pause_after_upticks:
                            print(">>> Pausing learning after %d upticks, validation loss rose to %.3f, best: %.3f" % (self.n_upticks, val_loss, self.best_val_loss))
                            self._pause_learning(epoch)

        return ret

    def _pause_learning(self, epoch):
        print("Setting eval interval to %d" % self.pause_eval_interval)
        self.model_learning_paused = True
        self.eval_interval = self.pause_eval_interval
        self.model._zero_kwinner_boost()

    def train_epoch(self, epoch):
        """
        Do one epoch of training and testing.

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
        total_loss = 0.0
        pcounts = {
            'total_samples': 0.0,
            'correct_samples': 0.0,
            'total_pred_loss': 0.0
        }

        bsz = self.batch_size

        read_out_tgt = []
        read_out_pred = []

        hidden = self.train_hidden_buffer[-1] if self.train_hidden_buffer else None
        if hidden is None:
            hidden = self._init_hidden(self.batch_size)

        for batch_idx, (inputs, targets, pred_targets, input_labels) in enumerate(
            self.train_loader
        ):
            # Inputs are of shape (batch, input_size)

            if inputs.size(0) > bsz:
                # Crop to smaller first epoch batch size
                inputs = inputs[:bsz]
                targets = targets[:bsz]
                pred_targets = pred_targets[:bsz]

            hidden = self._repackage_hidden(hidden)

            self.optimizer.zero_grad()
            if self.pred_optimizer:
                self.pred_optimizer.zero_grad()

            # Forward
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred_targets = pred_targets.to(self.device)

            output, hidden = self.model(inputs, hidden)

            x_b, pred_input = self._get_prediction_and_loss_inputs(hidden, output, inputs=inputs)

            self.train_output_buffer.append(output)
            self.train_hidden_buffer.append(hidden)

            # Loss
            # train_output_buffer holds last and present outputs
            outputs = self.train_output_buffer[0] if len(self.train_output_buffer) == 2 else None
            loss_targets = (inputs, x_b)
            loss = self._compute_loss(outputs, loss_targets)
            if loss is not None:
                total_loss += loss.item()
                if not self.model_learning_paused:                    
                    self._backward_and_optimize(loss)

            # Keep only latest batch states around
            self.train_output_buffer = self.train_output_buffer[-1:]
            self.train_hidden_buffer = self.train_hidden_buffer[-1:]

            pcounts, class_predictions, correct_arr = (
                self._do_prediction(
                    pred_input, pred_targets, pcounts, train=True,
                    batch_idx=batch_idx
                )
            )

            if epoch == 0 and batch_idx >= self.batches_in_first_epoch - 1:
                print("Breaking after %d batches in epoch %d" % (self.batches_in_first_epoch, epoch))
                break

        ret["stop"] = 0

        if self.eval_interval and (epoch - 1) % self.eval_interval == 0:

            # Evaluate each x epochs
            ret.update(self.eval_epoch(epoch))
            # if self.dataset_kind == "ptb" and epoch >= 12 and ret["val_pred_ppl"] > 380:
            #     ret["stop"] = 1

        train_time = time.time() - t1
        self._post_epoch(epoch)

        ret["train_loss"] = total_loss / (batch_idx + 1)
        if self.predictor:
            train_pred_loss = pcounts['total_pred_loss'] / (batch_idx + 1)
            ret["train_pred_ppl"] = lang_util.perpl(train_pred_loss)
            ret["train_pred_acc"] = 100 * pcounts['correct_samples'] / pcounts['total_samples']

        ret["epoch_time_train"] = train_time
        ret["epoch_time"] = time.time() - t1
        ret["learning_rate"] = self.learning_rate
        print(epoch, print_epoch_values(ret))
        return ret

    def _post_epoch(self, epoch):
        """
        The set of actions to do after each epoch of training: adjust learning
        rate, rezero sparse weights, and update boost strengths.
        """
        if self.pause_after_epochs and epoch == self.pause_after_epochs:
            self._pause_learning(epoch)
        self._adjust_learning_rate(epoch)
        if self.eval_interval_schedule:
            for step, new_interval in self.eval_interval_schedule:
                if step == epoch:
                    print(">> Changing eval interval to %d" % new_interval)
                    self.eval_interval = new_interval
        self.model._post_epoch(epoch)

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
            torch.onnx.export(
                self.model, dummy_input, self.graph_filename, verbose=True
            )

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
        pass


if __name__ == "__main__":
    print("Using torch version", torch.__version__)
    print("Torch device count=%d" % torch.cuda.device_count())

    config = {
        "data_dir": os.path.expanduser("~/nta/datasets"),
        "path": os.path.expanduser("~/nta/results"),
    }

    exp = RSMExperiment(config)
    exp.model_setup(config)
    for epoch in range(2):
        exp.train_epoch(epoch)
