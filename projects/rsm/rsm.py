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
from torch import nn

from nupic.torch.modules.k_winners import KWinners
from nupic.torch.modules.sparse_weights import SparseWeights
from util import activity_square, count_parameters, get_grad_printer


def topk_mask(x, k=2):
    """
    Simple functional version of KWinnersMask/KWinners since
    autograd function apparently not currently exportable by JIT
    """
    res = torch.zeros_like(x)
    topk, indices = x.topk(k, sorted=False)
    return res.scatter(-1, indices, 1)


class PredictiveProximalLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, predictive_activity):
        ctx.save_for_backward(input, weight, bias, predictive_activity)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, predictive_activity = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if predictive_activity is not None:
            avg_predictive = predictive_activity.mean(0)
            active_predictors = (avg_predictive > avg_predictive.mean()).float()
            grad_weight *= active_predictors.repeat(grad_weight.size(0), 1)

        return grad_input, grad_weight, grad_bias


class PredictiveProximalLinear(nn.Linear):

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(PredictiveProximalLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input, predictive_activity=None):
        return PredictiveProximalLinearFunction.apply(input, self.weight, self.bias, predictive_activity)


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
        )
        self._init_linear_weights()

    def forward(self, x):
        """
        Receive input as hidden memory state from RSM, batch
        x^B is with shape (batch_size, total_cells)

        Output is (batch_size, d_out)
        """
        x = self.layers(x)
        return x.view(-1, self.d_out)

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
        self.total_cells = kwargs['m'] * kwargs['n']
        self.batch_counter = 0
        self.hooks_registered = False
        eps_arr = self._parse_param_array(kwargs['eps'])
        k_winners_arr = self._parse_param_array(kwargs['k'])
        boost_strength_arr = self._parse_param_array(kwargs['boost_strength'])
        last_output_dim = None
        for i in range(n_layers):
            first_layer = i == 0
            top_layer = i == n_layers - 1
            if not first_layer:
                # Input to all layers above first is hidden state x_b from previous
                kwargs['d_in'] = last_output_dim
                # Output is of same dim as input (predictive autoencoder)
                kwargs['d_out'] = kwargs['d_in']
            if top_layer:
                kwargs['lateral_conn'] = kwargs['top_lateral_conn']
            kwargs['eps'] = eps_arr[i]
            kwargs['k'] = k_winners_arr[i]
            kwargs['boost_strength'] = boost_strength_arr[i]
            last_output_dim = kwargs['m'] * kwargs['n']
            self.add_module("RSM_%d" % (i+1), RSMLayer(**kwargs))
        print("Created RSMNet with %d layer(s)" % n_layers)

    def _parse_param_array(self, param_val):
        param_by_layer = param_val
        if not isinstance(param_by_layer, list):
            param_by_layer = [param_by_layer for x in range(self.n_layers)]
        return param_by_layer

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
            x_a_batch: (bsz, total_cells)
            hidden: Tuple (x_b, phi, psi), each Tensor (n_layers, bsz, total_cells)

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
        for (layer_name, layer), lay_phi, lay_psi in zip(self.named_children(), phi, psi):
            last_layer = lid == len(phi) - 1
            first_layer = lid == 0
            lay_x_b = x_b[lid]
            lay_x_above = x_b[lid+1] if not last_layer else None
            lay_x_below = x_b[lid-1] if not first_layer else None

            hidden_in = (lay_x_b, lay_x_above, lay_x_below, lay_phi, lay_psi)

            pred_output, hidden = layer(layer_input, hidden_in)

            layer_input = hidden[0]

            new_x_b.append(hidden[0])
            new_phi.append(hidden[1])
            new_psi.append(hidden[2])

            output_by_layer.append(pred_output)

            lid += 1

        new_hidden = (
            torch.stack(new_x_b),
            torch.stack(new_phi),
            torch.stack(new_psi)
        )

        self.batch_counter += 1  # Is this stored elsewhere?
        return (output_by_layer, new_hidden)

    def _plot_tensors(self, tuples, detailed=False, return_fig=False):
        """
        Plot first item in batch across multiple layers
        """
        n_tensors = len(tuples)
        fig, axs = plt.subplots(self.n_layers, n_tensors, dpi=144)
        for i, (label, val) in enumerate(tuples):
            for l in range(self.n_layers):
                layer_idx = self.n_layers - l - 1
                ax = axs[layer_idx][i] if n_tensors > 1 else axs[layer_idx]
                # Get layer's values (from either list or tensor)
                # Outputs can't be stored in tensors since dimension heterogeneous
                if isinstance(val, list):
                    if val[l] is None:
                        ax.set_visible(False)
                        t = None
                    else:
                        t = val[l].detach()[0]
                else:
                    t = val.detach()[l, 0]
                mod = list(self.children())[l]
                if t is not None:
                    size = t.numel()
                    is_cell_level = t.numel() == mod.total_cells and mod.n > 1
                    if is_cell_level:
                        ax.imshow(
                            t.view(mod.m, mod.n).t(),
                            origin="bottom",
                            extent=(0, mod.m - 1, 0, mod.n + 1),
                        )
                    else:
                        ax.imshow(activity_square(t))
                    tmin = t.min()
                    tmax = t.max()
                    tsum = t.sum()
                    title = "L%d %s" % (l+1, label)
                    if detailed:
                        title += " (%s, rng: %.3f-%.3f, sum: %.3f)" % (size, tmin, tmax, tsum)
                    ax.set_title(title)
        if return_fig:
            return fig
        else:
            plt.show()

    def _post_epoch(self, epoch):
        for mod in self.children():
            mod._post_epoch(epoch)

    def init_hidden(self, batch_size):
        param = next(self.parameters())
        x_b = param.new_zeros(
            (self.n_layers, batch_size, self.total_cells), dtype=torch.float32, requires_grad=False
        )
        phi = param.new_zeros(
            (self.n_layers, batch_size, self.total_cells), dtype=torch.float32, requires_grad=False
        )
        psi = param.new_zeros(
            (self.n_layers, batch_size, self.total_cells), dtype=torch.float32, requires_grad=False
        )
        return (x_b, phi, psi)


class RSMLayer(torch.nn.Module):
    def __init__(
        self,
        d_in=28 * 28,
        d_out=28 * 28,
        m=200,
        n=6,
        k=25,
        k_winner_cells=1,
        gamma=0.5,
        eps=0.5,
        activation_fn="tanh",
        embed_dim=0,
        vocab_size=0,
        decode_from_full_memory=False,
        debug_log_names=None,
        mask_shifted_pi=False,
        do_inhibition=True,
        boost_strat="rsm_inhibition",
        x_b_norm=False,
        boost_strength=1.0,
        mult_integration=False,
        boost_strength_factor=1.0,
        forget_mu=0.0,
        weight_sparsity=None,
        feedback_conn=False,
        input_bias=False,
        decode_bias=True,
        lateral_conn=True,
        tp_boosting=False,
        mem_gain=1.0,
        debug=False,
        visual_debug=False,
        fpartition=None,
        balance_part_winners=False,
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
        self.forget_mu = float(forget_mu)

        self.total_cells = m * n
        self.flattened = self.total_cells == self.m

        # Tweaks
        self.activation_fn = activation_fn
        self.decode_from_full_memory = decode_from_full_memory
        self.boost_strat = boost_strat
        self.x_b_norm = x_b_norm
        self.mask_shifted_pi = mask_shifted_pi
        self.do_inhibition = do_inhibition
        self.boost_strength = boost_strength
        self.boost_strength_factor = boost_strength_factor
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
        self.tp_boosting = tp_boosting
        self.lateral_conn = lateral_conn
        self.mem_gain = mem_gain

        self.debug = debug
        self.visual_debug = visual_debug
        self.debug_log_names = debug_log_names

        self._build_layers_and_kwinners()

        print("Created %s with %d trainable params" % (str(self), count_parameters(self)))

    def __str__(self):
        fp = ""
        if self.fpartition:
            fp = " partition=(%.2f,%.2f)" % (self.fpartition[0], self.fpartition[1])
        return "<RSMLayer m=%d n=%d k=%d d_in=%d eps=%.2f%s />" % (self.m, self.n, self.k, self.d_in, self.eps, fp)

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
        if self.fpartition:
            m_ff, m_int, m_rec = self._partition_sizes()
            # Partition memory into fpartition % FF & remainder recurrent
            self.linear_a = nn.Linear(self.d_in, m_ff, bias=self.input_bias)
            self.linear_b = nn.Linear(
                m_ff + m_rec, m_rec, bias=self.input_bias
            )  # Recurrent weights (per cell)
            if m_int:
                # Add two additional layers for integrating ff & rec input
                # NOTE: Testing int layer that gets only input from prior int (no ff)
                self.linear_a_int = nn.Linear(self.d_in, m_int, bias=self.input_bias)
                self.linear_b_int = nn.Linear(m_int, m_int, bias=self.input_bias)
        else:
            # Standard architecture, no partition
            if self.tp_boosting:
                self.linear_a = PredictiveProximalLinear(self.d_in, self.m,
                                                         bias=self.input_bias)
            else:
                self.linear_a = nn.Linear(
                    self.d_in, self.m, bias=self.input_bias
                )  # Input weights (shared per group / proximal)
            if self.lateral_conn:
                self.linear_b = nn.Linear(
                    self.total_cells, self.total_cells, bias=self.input_bias
                )  # Recurrent weights (per cell)
            if self.feedback_conn:
                # Linear layers for both recurrent input from above and below
                self.linear_b_above = nn.Linear(self.total_cells, self.total_cells,
                                                bias=self.input_bias)

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

        # Decode linear
        decode_d_in = self.total_cells if self.decode_from_full_memory else self.m
        self.linear_d = nn.Linear(decode_d_in, self.d_out, bias=self.decode_bias)

        self._init_linear_weights()

    def _init_linear_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                sd = 0.03
                mod.weight.data.normal_(0.0, sd)
                if mod.bias is not None:
                    mod.bias.data.normal_(0.0, sd)

    def _zero_sparse_weights(self):
        for mod in self.modules():
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
            duty_cycle_period=1000,
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
        return activity.view(activity.size(0), self.m, self.n).max(dim=2).values

    def _fc_weighted_ave(self, x_a, x_b, x_b_above=None, x_b_below=None):
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
                z_b = self.linear_b(x_b[:, -m_rec:])  # bsz x m_rec
                z_int_ff = self.linear_a_int(x_a)
                # NOTE: Testing from only int/rec portion of mem (no ff)
                z_int_rec = self.linear_b_int(x_b[:, m_ff: m_ff + m_int])
                z_int = z_int_ff * z_int_rec if self.mult_integration else z_int_ff + z_int_rec
                sigma = torch.cat((z_a, z_int, z_b), 1)  # bsz x m
            else:
                z_b = self.linear_b(x_b)  # bsz x m_rec
                z_log["z_b"] = z_b
                sigma = torch.cat((z_a, z_b), 1)  # bsz x m
        else:
            # Col activation from inputs repeated for each cell
            if self.tp_boosting and x_b_below is not None:
                predictive_activity = x_b_below
                z_a = self.linear_a(x_a, predictive_activity).repeat_interleave(self.n, 1)
            else:
                z_a = self.linear_a(x_a).repeat_interleave(self.n, 1)

            sigma = z_a
            z_log = {"z_a": z_a}

            # Cell activation from recurrent (lateral) input
            if self.lateral_conn:
                z_b = self.mem_gain * self.linear_b(x_b)
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

            self._debug_log({"m_pi": m_pi, "m_lambda": m_lambda})

            y_pre_act = m_pi * m_lambda * sigma

            del m_lambda

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
                winning_col_exp = torch.cat(winners, 1)
            else:
                winning_cols = self.kwinners_col(lambda_).view(bsz, self.m, 1)
                winning_col_exp = (
                    winning_cols.repeat(1, 1, self.n)
                    .view(bsz, self.total_cells)
                )

            self._debug_log({"winning_col_exp": winning_col_exp})
            premask_act = pi if self.mask_shifted_pi else sigma
            y_pre_act = m_pi * winning_col_exp * premask_act

        del m_pi

        return y_pre_act

    def _inhibited_winners(self, sigma, phi):
        """
        Compute y_lambda
        """
        # Apply inhibition to non-neg shifted sigma
        inh = (1 - phi) if self.do_inhibition else 1
        pi = inh * (sigma - sigma.min() + 1)
        self._debug_log({"pi": pi})

        pi = pi.detach()  # Prevent gradients from flowing through inhibition/masking

        y_pre_act = self._k_winners(sigma, pi)

        activation = {"tanh": torch.tanh, "relu": nn.functional.relu}[
            self.activation_fn
        ]
        y = activation(y_pre_act)  # 1 x total_cells

        return y

    def _decode_prediction(self, y):
        # Decode prediction (optionally through col-wise max bottleneck)
        decode_input = y if self.decode_from_full_memory else self._group_max(y)
        output = self.linear_d(decode_input)
        return output

    def _update_memory_and_inhibition(self, y, phi, psi):
        # Get updated psi (memory state), decay inactive
        psi = torch.max(psi * self.eps, y)

        # Update phi for next step (decay inhibition cells)
        phi = torch.max(phi * self.gamma, y)

        return (phi, psi)

    def forward(self, x_a_batch, hidden):
        """
        :param x_a_batch: Input batch of batch_size items from
        generating process (batch_size, d_in)
        :param hidden:
            x_b: (normalized) memory state at same layer at t-1
            x_b_above: memory state at layer above at t-1
            x_b_below: memory state at layer below at t-1
            phi: inhibition state
            psi: memory state at t-1

        Note that RSMLayer takes a 4-tuple that includes the feedback state
        from the layer above, x_c, while the RSMNet takes only 3-tuple for
        hidden state.
        """
        x_b, x_b_above, x_b_below, phi, psi = hidden

        phi, psi = self._do_forgetting(phi, psi)

        self._debug_log({"x_b": x_b, "x_a_batch": x_a_batch})

        sigma = self._fc_weighted_ave(x_a_batch, x_b,
                                      x_b_above=x_b_above,
                                      x_b_below=x_b_below)
        self._debug_log({"sigma": sigma})

        y = self._inhibited_winners(sigma, phi)

        pred_output = self._decode_prediction(y)
        self._debug_log({"y": y, "pred_output": pred_output})

        phi, psi = self._update_memory_and_inhibition(y, phi, psi)
        self._debug_log({"phi": phi, "psi": psi})

        # Update recurrent input / output x_b
        if self.x_b_norm:
            # Normalizing scalar (force sum(x_b) == 1)
            alpha = (psi.sum(dim=1) + 1e-9).unsqueeze(dim=1)
            x_b = psi / alpha
        else:
            x_b = psi

        hidden = (x_b, phi, psi)
        return (pred_output, hidden)


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
