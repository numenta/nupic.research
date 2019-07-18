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

    def forward(self, x):
        """
        Receive input as hidden memory state from RSM, batch
        x^B is with shape (batch_size, total_cells)

        Output is (batch_size, d_out)
        """
        x = self.layers(x)
        return x.view(-1, self.d_out)


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
        dropout_p=0.0,
        decode_from_full_memory=False,
        debug_log_names=None,
        mask_shifted_pi=False,
        do_inhibition=True,
        boost_strat="rsm_inhibition",
        pred_gain=1.0,
        x_b_norm=False,
        boost_strength=1.0,
        mult_integration=False,
        boost_strength_factor=1.0,
        forget_mu=0.0,
        weight_sparsity=None,
        debug=False,
        visual_debug=False,
        use_bias=True,
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
        self.dropout_p = float(dropout_p)
        self.forget_mu = float(forget_mu)

        self.total_cells = m * n
        self.flattened = self.total_cells == self.m

        # Tweaks
        self.activation_fn = activation_fn
        self.decode_from_full_memory = decode_from_full_memory
        self.boost_strat = boost_strat
        self.pred_gain = pred_gain
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

        self.debug = debug
        self.visual_debug = visual_debug
        self.debug_log_names = debug_log_names

        self.dropout = nn.Dropout(p=self.dropout_p)

        self._build_input_layers_and_kwinners(use_bias=use_bias)

        decode_d_in = self.total_cells if self.decode_from_full_memory else m
        self.linear_d = nn.Linear(
            decode_d_in, d_out, bias=use_bias
        )  # Decoding through bottleneck

        print("Created model with %d trainable params" % count_parameters(self))

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

    def _zero_sparse_weights(self):
        self.linear_a.rezero_weights()
        self.linear_b.rezero_weights()

    def _partition_sizes(self):
        pct_ff, pct_rec = self.fpartition
        m_ff = int(pct_ff * self.m)
        m_rec = int(pct_rec * self.m)
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

    def _build_input_layers_and_kwinners(self, use_bias=True):

        if self.fpartition:
            m_ff, m_int, m_rec = self._partition_sizes()
            # Partition memory into fpartition % FF & remainder recurrent
            self.linear_a = nn.Linear(self.d_in, m_ff + m_int, bias=use_bias)
            self.linear_b = nn.Linear(
                self.total_cells, m_rec + m_int, bias=use_bias
            )  # Recurrent weights (per cell)
        else:
            # Standard architecture, no partition
            self.linear_a = nn.Linear(
                self.d_in, self.m, bias=use_bias
            )  # Input weights (shared per group / proximal)
            self.linear_b = nn.Linear(
                self.total_cells, self.total_cells, bias=use_bias
            )  # Recurrent weights (per cell)

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

    def _post_epoch(self, epoch):
        # Update boost strength of any KWinners modules
        for mod in self.modules:
            if hasattr(mod, 'update_boost_strength'):
                mod.update_boost_strength()

    def _register_hooks(self):
        """Utility function to call retain_grad and Pytorch's register_hook
        in a single line
        """
        for label, t in [
            # ('y', self.y),
            # ('sigma', self.sigma),
            ("linear_b", self.linear_b.weight)
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

    def _fc_weighted_ave(self, x_a, x_b):
        """
        Compute sigma (weighted sum for each cell j in group i (mxn))
        """
        if self.fpartition:
            m_ff, m_int, m_rec = self._partition_sizes()
            sigma = torch.zeros_like(x_b)
            # Integrate partitioned memory.
            # Pack as 1xm: [ ... m_ff ... ][ ... m_int ... ][ ... m_rec ... ]
            # If m_int non-zero, these cells receive sum of FF & recurrent input
            z_a = self.linear_a(x_a)  # bsz x (m_ff+m_int)
            z_b = self.linear_b(self.dropout(x_b))  # bsz x (m_rec+m_int)
            sigma[:, : m_ff + m_int] += z_a
            sigma[:, -(m_rec + m_int):] += z_b
        else:
            # Col activation from inputs repeated for each cell
            z_a = self.linear_a(x_a).repeat_interleave(self.n, 1)
            # Cell activation from recurrent input (dropped out)
            z_b = self.linear_b(self.dropout(x_b))
            self._debug_log({"z_a": z_a, "z_b": z_b})
            if self.mult_integration:
                sigma = z_a * z_b
            else:
                sigma = z_a + z_b
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
                    winners_int = self.kwinners_int(lambda_[:, m_ff:m_ff + m_int])
                    winners.append(winners_int)
                if self.kwinners_rec is not None:
                    winners_rec = self.kwinners_rec(lambda_[:, -m_rec:])
                    winners.append(winners_rec)
                winning_columns = torch.cat(winners, 1)
            else:
                winning_columns = (
                    self.kwinners_col(lambda_)
                    .view(bsz, self.m, 1)
                    .repeat(1, 1, self.n)
                    .view(bsz, self.total_cells)
                )
            self._debug_log({"winning_columns": winning_columns})
            premask_act = pi if self.mask_shifted_pi else sigma
            y_pre_act = m_pi * winning_columns * premask_act

        del m_pi

        return y_pre_act

    def _inhibited_masking_and_prediction(self, sigma, phi):
        """
        Compute y_lambda
        """
        # Apply inhibition to non-neg shifted sigma
        inh = (1 - phi) if self.do_inhibition else 1
        pi = inh * (sigma - sigma.min() + 1)
        self._debug_log({"pi": pi})

        pi = pi.detach()  # Prevent gradients from flowing through inhibition/masking

        y_pre_act = self._k_winners(sigma, pi)

        # Mask-based sparsening
        activation = {"tanh": torch.tanh, "relu": nn.functional.relu}[
            self.activation_fn
        ]
        y = activation(y_pre_act)  # 1 x total_cells

        # Decode prediction through group-wise max bottleneck
        decode_input = y if self.decode_from_full_memory else self._group_max(y)
        output = self.linear_d(decode_input)

        return (y, output)

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
        """
        x_b, phi, psi = hidden

        phi, psi = self._do_forgetting(phi, psi)

        self._debug_log({"x_b": x_b})

        self._debug_log({"x_a_batch": x_a_batch})

        sigma = self._fc_weighted_ave(x_a_batch, x_b)
        self._debug_log({"sigma": sigma})

        y, pred_output = self._inhibited_masking_and_prediction(sigma, phi)
        self._debug_log({"y": y, "pred_output": pred_output})

        phi, psi = self._update_memory_and_inhibition(y, phi, psi)
        self._debug_log({"phi": phi, "psi": psi})

        # Update recurrent input / output x_b
        if self.x_b_norm:
            # Normalizing scalar (force sum(x_b) == 1)
            alpha = (psi.sum(dim=1) + 1e-9).unsqueeze(dim=1)
            x_b = self.pred_gain * psi / alpha
        else:
            x_b = self.pred_gain * psi

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
