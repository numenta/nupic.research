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

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# from nupic.torch.modules.k_winners import KWinners
from k_winners import KWinners
from nupic.torch.modules.sparse_weights import SparseWeights
from util import activity_square, count_parameters, get_grad_printer
from active_dendrite import ActiveDendriteLayer


def topk_mask(x, k=2):
    """
    Simple functional version of KWinnersMask/KWinners since
    autograd function apparently not currently exportable by JIT
    """
    res = torch.zeros_like(x)
    topk, indices = x.topk(k, sorted=False)
    return res.scatter(-1, indices, 1)


class RSMPredictor(torch.nn.Module):
    def __init__(self, d_in=28 * 28, d_out=10, hidden_size=20):
        """

        """
        super(RSMPredictor, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_size = hidden_size

        self.layers = nn.Sequential(
            nn.Linear(self.d_in, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.d_out),
            nn.LeakyReLU(),
            nn.Softmax(dim=1),
        )
        self._init_linear_weights()

    def forward(self, x):
        """
        Receive input as hidden memory state from RSM, batch
        x^B is with shape (batch_size, total_cells)

        Output is two tensors of shape (batch_size, d_out) being distribution and logits respectively.
        """
        x1 = None
        x2 = x
        for layer in self.layers:
            x1 = x2
            x2 = layer(x1)

        # Return multiple outputs
        # https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
        logits = x1.view(-1, self.d_out)
        distribution = x2.view(-1, self.d_out)
        return distribution, logits

    def _init_linear_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                sd = 0.03
                mod.weight.data.normal_(0.0, sd)
                if mod.bias is not None:
                    mod.bias.data.normal_(0.0, sd)


class RSMNet(torch.nn.Module):
    def __init__(self, n_layers=1, **kwargs):
        super(RSMNet, self).__init__()
        self.n_layers = n_layers
        self.hooks_registered = False

        eps_arr = self._parse_param_array(kwargs["eps"])
        k_winners_arr = self._parse_param_array(kwargs["k"])
        boost_strength_arr = self._parse_param_array(kwargs["boost_strength"])
        duty_cycle_period_arr = self._parse_param_array(
            kwargs.get("duty_cycle_period", 1000)
        )
        m_arr = self._parse_param_array(kwargs["m"])
        n_arr = self._parse_param_array(kwargs["n"])
        last_output_dim = None
        self.total_cells = []
        for i in range(n_layers):
            first_layer = i == 0
            top_layer = i == n_layers - 1
            if not first_layer:
                kwargs["d_in"] = last_output_dim
                # Output is of same dim as input (predictive autoencoder)
                kwargs["d_out"] = kwargs["d_in"]
            if not top_layer:
                kwargs["d_above"] = m_arr[i + 1] * n_arr[i + 1]
            if kwargs.get("lateral_conn", True):
                if top_layer:
                    kwargs["lateral_conn"] = kwargs.get("top_lateral_conn", False)
            kwargs["eps"] = eps_arr[i]
            kwargs["m"] = m_arr[i]
            kwargs["n"] = n_arr[i]
            kwargs["k"] = k_winners_arr[i]
            kwargs["boost_strength"] = boost_strength_arr[i]
            kwargs["duty_cycle_period"] = duty_cycle_period_arr[i]
            self.total_cells.append(kwargs["m"] * kwargs["n"])
            last_output_dim = kwargs["m"] * kwargs["n"]
            self.add_module("RSM_%d" % (i + 1), RSMLayer(**kwargs))

        print("Created RSMNet with %d layer(s)" % n_layers)

    def _parse_param_array(self, param_val):
        param_by_layer = param_val
        if not isinstance(param_by_layer, list):
            param_by_layer = [param_by_layer for x in range(self.n_layers)]
        return param_by_layer

    def _zero_sparse_weights(self):
        for mod in self.children():
            mod._zero_sparse_weights()

    def _zero_kwinner_boost(self):
        # Zero KWinner boost strengths since learning in RSM is pausing
        for layer in self.children():
            for mod in layer.children():
                if isinstance(mod, KWinners) and mod.boost_strength_factor < 1.0:
                    print("Zeroing boost strength for %s" % mod)
                    mod.boost_strength = 0.0

    def _register_hooks(self):
        if not self.hooks_registered:
            for mod in self.children():
                mod._register_hooks()
        self.hooks_registered = True

    def forward(self, x_a_batch, hidden):
        """
        Each layer takes input (image batch from time sequence for first layer,
        batch of hidden states from prior layer otherwise), and generates both:
            - a prediction for the next input it will see
            - a hidden state which is passed to the next layer

        Arguments:
            x_a_batch: (bsz, d_in)
            hidden: Tuple (x_b, phi, psi), each Tensor (n_layers, bsz, total_cells)
                - x_b is (possibly normalized) winners without hysteresis/decayed memory

        Returns:
            output_by_layer: List of tensors (n_layers, bsz, dim (total_cells or d_in for first layer))
            new_hidden: Tuple of tensors (n_layers, bsz, total_cells)
        """
        output_by_layer = []

        new_x_b = []
        new_phi = []
        new_psi = []

        x_b, phi, psi = hidden
        layer_input = x_a_batch

        lid = 0
        for (layer_name, layer), lay_phi, lay_psi in zip(
            self.named_children(), phi, psi
        ):
            last_layer = lid == len(phi) - 1
            lay_above = list(self.children())[lid + 1] if not last_layer else None
            lay_x_b = x_b[lid]
            lay_x_above = x_b[lid + 1] if not last_layer else None

            # Update memory psi with prior step winners and apply decay as per config
            if lay_x_above is not None:
                psi_above = psi[lid + 1]
                lay_x_above = lay_above._decay_memory(psi_above, lay_x_above)
            if lay_x_b is not None:
                lay_x_b = layer._decay_memory(lay_psi, lay_x_b)

            hidden_in = (lay_x_b, lay_x_above, lay_phi, lay_psi)

            pred_output, hidden = layer(layer_input, hidden_in)

            # If layers > 1, higher layers predict lower layer's phi (TOOD: or should be x_b?)
            # phi has hysteresis (if decay active), x_b is just winners
            layer_input = hidden[1]  # phi

            new_x_b.append(hidden[0])
            new_phi.append(hidden[1])
            new_psi.append(hidden[2])

            output_by_layer.append(pred_output)

            lid += 1

        new_hidden = (new_x_b, new_phi, new_psi)

        return (output_by_layer, new_hidden)

    def _post_train_epoch(self, epoch):
        for mod in self.children():
            mod._post_epoch(epoch)

    def init_hidden(self, batch_size):
        param = next(self.parameters())
        x_b = [
            param.new_zeros((batch_size, tc), dtype=torch.float32, requires_grad=False)
            for tc in self.total_cells
        ]
        phi = [
            param.new_zeros((batch_size, tc), dtype=torch.float32, requires_grad=False)
            for tc in self.total_cells
        ]
        psi = [
            param.new_zeros((batch_size, tc), dtype=torch.float32, requires_grad=False)
            for tc in self.total_cells
        ]
        return (x_b, phi, psi)


class RSMLayer(torch.nn.Module):
    ACT_FNS = {"tanh": torch.tanh, "relu": F.relu, "sigmoid": torch.sigmoid}

    def __init__(
        self,
        d_in=28 * 28,
        d_out=28 * 28,
        d_above=None,
        m=200,
        n=6,
        k=25,
        k_winner_cells=1,
        gamma=0.5,
        eps=0.5,
        activation_fn="tanh",
        decode_activation_fn=None,
        embed_dim=0,
        vocab_size=0,
        decode_from_full_memory=False,
        debug_log_names=None,
        boost_strat="rsm_inhibition",
        x_b_norm=False,
        boost_strength=1.0,
        duty_cycle_period=1000,
        mult_integration=False,
        boost_strength_factor=1.0,
        forget_mu=0.0,
        weight_sparsity=None,
        feedback_conn=False,
        input_bias=False,
        decode_bias=True,
        lateral_conn=True,
        col_output_cells=False,
        debug=False,
        visual_debug=False,
        fpartition=None,
        balance_part_winners=False,
        trainable_decay=False,
        trainable_decay_rec=False,
        max_decay=1.0,
        mem_floor=0.0,
        additive_decay=False,
        stoch_decay=False,
        stoch_k_sd=0.0,
        rec_active_dendrites=0,
        **kwargs,
    ):
        """
        This class includes an attempted replication of the Recurrent Sparse Memory
        architecture suggested by by
        [Rawlinson et al 2019](https://arxiv.org/abs/1905.11589).

        Parameters allow experimentation with a wide array of adjustments to this model,
        both minor and major. Classes of models tested include:

        * "Adjusted" model with k-winners and column boosting, 2 cell winners,
            no inhibition
        * "Flattened" model with 1 cell per column, 1000 cols, 25 winners
            and multiplicative integration of FF & recurrent input
        * "Flat Partitioned" model with 120 winners, and cells partitioned into three
            functional types: ff only, recurrent only, and optionally a region that
            integrates both.

        :param d_in: Dimension of input
        :param m: Number of groups/columns
        :param n: Cells per group/column
        :param k: # of groups/columns to win in topk() (sparsity)
        :param k_winner_cells: # of winning cells per column
        :param gamma: Inhibition decay rate (0-1)
        :param eps: Integrated encoding decay rate (0-1)

        """
        super(RSMLayer, self).__init__()
        self.k = int(k)
        self.k_winner_cells = k_winner_cells
        self.m = m
        self.n = n
        self.gamma = gamma
        self.eps = eps
        self.d_in = d_in
        self.d_out = d_out
        self.d_above = d_above
        self.forget_mu = float(forget_mu)

        self.total_cells = m * n
        self.flattened = self.total_cells == self.m

        # Tweaks
        self.activation_fn = activation_fn
        self.decode_activation_fn = decode_activation_fn
        self.decode_from_full_memory = decode_from_full_memory
        self.boost_strat = boost_strat
        self.x_b_norm = x_b_norm
        self.boost_strength = boost_strength
        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period
        self.mult_integration = mult_integration
        self.fpartition = fpartition
        if isinstance(self.fpartition, float):
            # Handle simple single-param FF-percentage only
            # If fpartition is list, interpreted as [ff_pct, rec_pct]
            self.fpartition = [self.fpartition, 1.0 - self.fpartition]
        self.balance_part_winners = balance_part_winners
        self.weight_sparsity = weight_sparsity
        self.feedback_conn = feedback_conn
        self.input_bias = input_bias
        self.decode_bias = decode_bias
        self.lateral_conn = lateral_conn
        self.trainable_decay = trainable_decay
        self.trainable_decay_rec = trainable_decay_rec
        self.max_decay = max_decay
        self.additive_decay = additive_decay
        self.stoch_decay = stoch_decay
        self.col_output_cells = col_output_cells
        self.stoch_k_sd = stoch_k_sd
        self.rec_active_dendrites = rec_active_dendrites
        self.mem_floor = mem_floor

        self.debug = debug
        self.visual_debug = visual_debug
        self.debug_log_names = debug_log_names

        self._build_layers_and_kwinners()

        if self.additive_decay:
            decay_init = torch.ones(self.total_cells, dtype=torch.float32).uniform_(
                -3.0, 3.0
            )
        elif self.stoch_decay:
            # Fixed random decay rates, test with trainable_decay = False
            decay_init = torch.ones(self.total_cells, dtype=torch.float32).uniform_(
                -3.0, 3.0
            )
        else:
            decay_init = self.eps * torch.ones(self.total_cells, dtype=torch.float32)
        self.decay = nn.Parameter(decay_init, requires_grad=self.trainable_decay)
        self.register_parameter("decay", self.decay)
        self.learning_iterations = 0
        self.register_buffer("duty_cycle", torch.zeros(self.total_cells))

        print(
            "Created %s with %d trainable params" % (str(self), count_parameters(self))
        )

    def __str__(self):
        fp = ""
        if self.fpartition:
            fp = " partition=(%.2f,%.2f)" % (self.fpartition[0], self.fpartition[1])
        return "<RSMLayer m=%d n=%d k=%d d_in=%d eps=%.2f%s />" % (
            self.m,
            self.n,
            self.k,
            self.d_in,
            self.eps,
            fp,
        )

    def _debug_log(self, tensor_dict, truncate_len=400):
        if self.debug:
            for name, t in tensor_dict.items():
                if not self.debug_log_names or name in self.debug_log_names:
                    _type = type(t)
                    if _type in [int, float, bool]:
                        size = "-"
                    else:
                        size = t.size()
                        _type = t.dtype
                        if t.numel() > truncate_len:
                            t = "..truncated.."
                    print([name, t, size, _type])
        if self.visual_debug:
            for name, t in tensor_dict.items():
                if not self.debug_log_names or name in self.debug_log_names:
                    if isinstance(t, torch.Tensor):
                        t = t.detach().squeeze()
                        if t.dim() == 1:
                            t = t.flatten()
                            size = t.numel()
                            is_cell_level = t.numel() == self.total_cells and self.n > 1
                            if is_cell_level:
                                plt.imshow(
                                    t.view(self.m, self.n).t(),
                                    origin="bottom",
                                    extent=(0, self.m - 1, 0, self.n),
                                )
                            else:
                                plt.imshow(activity_square(t))
                            tmin = t.min()
                            tmax = t.max()
                            tsum = t.sum()
                            plt.title(
                                "%s (%s, rng: %.3f-%.3f, sum: %.3f)"
                                % (name, size, tmin, tmax, tsum)
                            )
                            plt.show()

    def _build_layers_and_kwinners(self):
        self.sparse_mods = []
        if self.fpartition:
            m_ff, m_int, m_rec = self._partition_sizes()
            # Partition memory into fpartition % FF & remainder recurrent
            self.linear_a = nn.Linear(self.d_in, m_ff, bias=self.input_bias)
            self.linear_b = nn.Linear(
                self.total_cells, m_rec, bias=self.input_bias
            )  # Recurrent weights (per cell)
            if m_int:
                # Add two additional layers for integrating ff & rec input
                # NOTE: Testing int layer that gets only input from prior int (no ff)
                self.linear_a_int = nn.Linear(self.d_in, m_int, bias=self.input_bias)
                self.linear_b_int = nn.Linear(
                    self.total_cells, m_int, bias=self.input_bias
                )
        else:
            # Standard architecture, no partition
            self.linear_a = nn.Linear(
                self.d_in, self.m, bias=self.input_bias
            )  # Input weights (shared per group / proximal)
            if self.lateral_conn:
                d1 = d2 = self.total_cells
                if self.col_output_cells:
                    d1 += self.m  # One output per column
                # Recurrent weights (per cell)
                if self.rec_active_dendrites:
                    sparsity = 0.3
                    self.linear_b = ActiveDendriteLayer(
                        d1,
                        n_cells=d2,
                        n_dendrites=self.rec_active_dendrites,
                        sparsity=sparsity,
                    )
                    if sparsity:
                        self.sparse_mods.append(self.linear_b.linear_dend)
                else:
                    self.linear_b = nn.Linear(d1, d2, bias=self.input_bias)

            if self.feedback_conn:
                # Linear layers for both recurrent input from above and below
                self.linear_b_above = nn.Linear(
                    self.d_above, self.total_cells, bias=self.input_bias
                )

        pct_on = self.k / self.m
        if self.fpartition and self.balance_part_winners:
            # Create a kwinners module for each partition each with specified
            # size but same pct on (balanced).
            self.kwinners_ff = self.kwinners_rec = self.kwinners_int = None
            if m_ff:
                self.kwinners_ff = self._build_kwinner_mod(m_ff, pct_on)
            if m_int:
                self.kwinners_int = self._build_kwinner_mod(m_int, pct_on)
            if m_rec:
                self.kwinners_rec = self._build_kwinner_mod(m_rec, pct_on)
        else:
            # We need only a single kwinners to run on full memory
            self.kwinners_col = self._build_kwinner_mod(self.m, pct_on)
        if self.weight_sparsity is not None:
            self.linear_a = SparseWeights(self.linear_a, self.weight_sparsity)
            self.linear_b = SparseWeights(self.linear_b, self.weight_sparsity)
            self.sparse_mods.extend([self.linear_a, self.linear_b])

        # Decode linear
        decode_d_in = self.total_cells if self.decode_from_full_memory else self.m
        self.linear_d = nn.Linear(decode_d_in, self.d_out, bias=self.decode_bias)

        if self.trainable_decay_rec:
            self.linear_decay_rec = nn.Linear(
                self.total_cells, self.total_cells, bias=True
            )

        self._init_linear_weights()

    def _init_linear_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                sd = 0.03
                mod.weight.data.normal_(0.0, sd)
                if mod.bias is not None:
                    mod.bias.data.normal_(0.0, sd)

    def _zero_sparse_weights(self):
        for mod in self.sparse_mods:
            if isinstance(mod, SparseWeights):
                mod.rezero_weights()

    def _partition_sizes(self):
        pct_ff, pct_rec = self.fpartition
        m_ff = int(round(pct_ff * self.m))
        m_rec = int(round(pct_rec * self.m))
        m_int = self.m - m_ff - m_rec
        return (m_ff, m_int, m_rec)

    def _build_kwinner_mod(self, m, pct_on):
        return KWinners(
            m,
            pct_on,
            boost_strength=self.boost_strength,
            duty_cycle_period=self.duty_cycle_period,
            k_inference_factor=1.0,
            stoch_sd=self.stoch_k_sd,
            boost_strength_factor=self.boost_strength_factor,
        )

    def _post_epoch(self, epoch):
        # Update boost strength of any KWinners modules
        for mod in self.modules():
            if hasattr(mod, "update_boost_strength"):
                mod.update_boost_strength()

    def _register_hooks(self):
        """Utility function to call retain_grad and Pytorch's register_hook
        in a single line
        """
        for label, t in [
            # ('y', self.y),
            # ('sigma', self.sigma),
            ("linear_b grad", self.linear_b.weight)
        ]:
            t.retain_grad()
            t.register_hook(get_grad_printer(label))

    def _decay_memory(self, psi_last, x_b):
        if self.trainable_decay_rec:
            decay_param = self.max_decay * torch.sigmoid(
                self.linear_decay_rec(psi_last)
            )
        elif self.trainable_decay:
            decay_param = self.max_decay * torch.sigmoid(self.decay)
        else:
            decay_param = self.eps

        updated = decay_param * psi_last
        if self.mem_floor:
            updated[updated <= self.mem_floor] = 0.0
        memory = torch.max(updated, x_b)
        return memory

    def _do_forgetting(self, phi, psi):
        bsz = phi.size(0)
        if self.training and self.forget_mu > 0:
            keep_idxs = torch.rand(bsz) > self.forget_mu
            mask = torch.zeros_like(phi)
            mask[keep_idxs, :] = 1
            phi = phi * mask
            psi = psi * mask
        return (phi, psi)

    def _group_max(self, activity):
        """
        :param activity: activity vector (bsz x total_cells)

        Returns max cell activity in each group
        """
        return activity.view(-1, self.m, self.n).max(dim=2).values

    def _fc_weighted_ave(self, x_a, x_b, x_b_above=None):
        """
        Compute sigma (weighted sum for each cell j in group i (mxn))
        """
        if self.fpartition:
            m_ff, m_int, m_rec = self._partition_sizes()
            sigma = torch.zeros_like(x_b)
            # Integrate partitioned memory.
            # Pack as 1xm: [ ... m_ff ... ][ ... m_int ... ][ ... m_rec ... ]
            # If m_int non-zero, these cells receive sum of FF & recurrent input
            z_a = self.linear_a(x_a)  # bsz x (m_ff)
            z_log = {"z_a": z_a}
            if m_int:
                z_b = self.linear_b(x_b)  # bsz x m_rec
                z_int_ff = self.linear_a_int(x_a)
                # NOTE: Testing from only int/rec portion of mem (no ff)
                z_int_rec = self.linear_b_int(x_b)
                z_int = (
                    z_int_ff * z_int_rec
                    if self.mult_integration
                    else z_int_ff + z_int_rec
                )
                sigma = torch.cat((z_a, z_int, z_b), 1)  # bsz x m
            else:
                z_b = self.linear_b(x_b)  # bsz x m_rec
                z_log["z_b"] = z_b
                sigma = torch.cat((z_a, z_b), 1)  # bsz x m
        else:
            # Col activation from inputs repeated for each cell
            z_a = self.linear_a(x_a).repeat_interleave(self.n, 1)

            sigma = z_a
            z_log = {"z_a": z_a}

            # Cell activation from recurrent (lateral) input
            if self.lateral_conn:
                z_b_in = x_b
                if self.col_output_cells:
                    z_b_in = torch.cat((z_b_in, self._group_max(x_b)), dim=1)
                z_b = self.linear_b(z_b_in)
                sigma = sigma * z_b if self.mult_integration else sigma + z_b
                z_log["z_b"] = z_b
            # Activation from recurrent (feedback) input
            if self.feedback_conn:
                if x_b_above is not None:
                    # Cell activation from recurrent input from layer above (apical)
                    z_b_above = self.linear_b_above(x_b_above)
                    z_log["z_b_above"] = z_b_above
                    if self.mult_integration:
                        sigma = sigma * z_b_above
                    else:
                        sigma = sigma + z_b_above
        self._debug_log(z_log)

        return sigma  # total_cells

    def _update_duty_cycle(self, winners):
        """
        For tracking layer entropy (across both inhibition/boosting approaches)
        """
        batch_size = winners.shape[0]
        self.learning_iterations += batch_size
        period = min(1000, self.learning_iterations)
        self.duty_cycle.mul_(period - batch_size)
        self.duty_cycle.add_(winners.gt(0).sum(dim=0, dtype=torch.float))
        self.duty_cycle.div_(period)

    def _k_winners(self, sigma, pi):
        bsz = pi.size(0)

        # Group-wise max pooling
        if not self.flattened:
            lambda_ = self._group_max(pi)
        else:
            lambda_ = pi

        # Cell-level mask: Make a bsz x total_cells binary mask of top 1 cell / column
        if self.n == self.k_winner_cells:
            # Usually just in flattened case, no need to choose winners
            m_pi = torch.ones(bsz, self.total_cells, device=sigma.device)
        else:
            mask = topk_mask(pi.view(bsz * self.m, self.n), self.k_winner_cells)
            m_pi = mask.view(bsz, self.total_cells).detach()

        if self.boost_strat == "rsm_inhibition":
            # Standard RSM-style inhibition via phi matrix

            self._debug_log({"lambda_": lambda_})

            # Column-level mask: Make a bsz x total_cells binary mask of top k columns
            mask = topk_mask(lambda_, self.k)
            m_lambda = (
                mask.view(bsz, self.m, 1)
                .repeat(1, 1, self.n)
                .view(bsz, self.total_cells)
                .detach()
            )
            col_winners = m_lambda

            self._debug_log({"m_pi": m_pi, "m_lambda": m_lambda})

            y_pre_act = m_pi * m_lambda * sigma

        elif self.boost_strat == "col_boosting":
            # HTM style boosted k-winner
            if self.balance_part_winners and self.fpartition:
                m_ff, m_int, m_rec = self._partition_sizes()
                winners = []
                if self.kwinners_ff is not None:
                    winners_ff = self.kwinners_ff(lambda_[:, :m_ff])
                    winners.append(winners_ff)
                if self.kwinners_int is not None:
                    winners_int = self.kwinners_int(lambda_[:, m_ff : m_ff + m_int])
                    winners.append(winners_int)
                if self.kwinners_rec is not None:
                    winners_rec = self.kwinners_rec(lambda_[:, -m_rec:])
                    winners.append(winners_rec)
                m_lambda = (torch.cat(winners, 1).abs() > 0).float()
            else:
                winning_cols = (
                    self.kwinners_col(lambda_).view(bsz, self.m, 1).abs() > 0
                ).float()
                m_lambda = winning_cols.repeat(1, 1, self.n).view(bsz, self.total_cells)
                col_winners = winning_cols

            self._debug_log({"m_lambda": m_lambda})
            y_pre_act = m_pi * m_lambda * sigma

        self._update_duty_cycle(col_winners.squeeze())

        del m_pi
        del m_lambda

        return y_pre_act

    def _inhibited_winners(self, sigma, phi):
        """
        Compute y_lambda
        """
        # Apply inhibition to non-neg shifted sigma
        inh = (1 - phi) if self.boost_strat == "rsm_inhibition" else 1
        pi = inh * (sigma - sigma.min() + 1)
        self._debug_log({"pi": pi})

        pi = pi.detach()  # Prevent gradients from flowing through inhibition/masking

        y_pre_act = self._k_winners(sigma, pi)

        activation = RSMLayer.ACT_FNS[self.activation_fn]
        y = activation(y_pre_act)  # 1 x total_cells

        return y

    def _update_memory_and_inhibition(self, y, phi, psi, x_b=None):
        """
        Decay memory and inhibition tensors
        """

        # Set psi to x_b, which includes decayed prior state (see RSMNet.forward)
        psi = x_b

        # Update phi for next step (decay inhibition cells)
        phi = torch.max(phi * self.gamma, y)

        return (phi, psi)

    def _decode_prediction(self, y):
        if self.decode_from_full_memory:
            decode_input = y
        else:
            decode_input = self._group_max(y)
        output = self.linear_d(decode_input)
        if self.decode_activation_fn:
            activation = RSMLayer.ACT_FNS[self.decode_activation_fn]
            output = activation(output)
        return output

    def forward(self, x_a_batch, hidden):
        """
        :param x_a_batch: Input batch of batch_size items from
        generating process (batch_size, d_in)
        :param hidden:
            x_b: Memory at same layer at t-1 (possibly with decayed/hysteresis memory from prior time steps)
            x_b_above: memory state at layer above at t-1
            phi: inhibition state (used only for boost_strat=='rsm_inhibition')
            psi: memory state at t-1, inclusive of hysteresis / ramped values if applicable

        Note that RSMLayer takes a 4-tuple that includes the feedback state
        from the layer above, x_c, while the RSMNet takes only 3-tuple for
        hidden state.
        """
        x_b, x_b_above, phi, psi = hidden
        x_b_in = x_b.clone()

        phi, psi = self._do_forgetting(phi, psi)

        self._debug_log({"x_b": x_b, "x_a_batch": x_a_batch})

        sigma = self._fc_weighted_ave(x_a_batch, x_b, x_b_above=x_b_above)
        self._debug_log({"sigma": sigma})

        y = self._inhibited_winners(sigma, phi)

        phi, psi = self._update_memory_and_inhibition(y, phi, psi, x_b=x_b_in)
        self._debug_log({"phi": phi, "psi": psi})

        output = self._decode_prediction(y)
        self._debug_log({"y": y, "output": output})

        # Update recurrent input / output x_b
        if self.x_b_norm:
            # Normalizing scalar (force sum(x_b) == 1)
            alpha_y = (y.sum(dim=1) + 1e-9).unsqueeze(dim=1)
            x_b = y / alpha_y
        else:
            x_b = y

        hidden = (x_b, phi, psi)
        return (output, hidden)


if __name__ == "__main__":
    batch_size, d_in = 50, 64

    x = torch.randn(batch_size, d_in)
    y = torch.randn(batch_size, d_in)

    model = RSMLayer(d_in)

    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(500):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
