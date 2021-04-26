import torch
import torch.nn as nn

from nupic.research.frameworks.greedy_infomax.utils.model_utils import (
    makeDeltaOrthogonal,
)


class BilinearInfo(nn.Module):
    """
    From the Greedy InfoMax paper, a module which estimates the mutual information
    between two representations through a bilinear model.

    Recall the loss for Greedy InfoMax:

    f(z_{t+k}, z_{t}) = exp(z_{t+k} W_k z_{t})

    L = - ∑_{k} E[log (f(z_{t+k}, z_{t}) / ∑_{z_j} f(z_{j}, z_{t})]

    This model computes:

        log f(z_{t+k}, z_{t}) = z_{t+k} W_k z_{t}

    for each prediction step k along with a specified number of random contrastive
    samples.

    After computing this value for all necessary positive and contrastive samples,
    the positive examples are at index 0 and the rest are indices 1:n, where n is
    the number of contrastive samples per positive sample. Therefore, the the loss
    simply becomes nn.CrossEntropy with the "class" index set to 0:

        loss(x,class)= −log( exp(x[class] / (∑_j exp(x[j]) )

    :param negative_samples: number of negative samples to contrast per positive sample.
    :param k_predictions: number of prediction steps to compare positive examples.
                          For example, if k_predictions is 5 and skip_step is 1,
                          then this module will compare z_{t} with z_{t+2}...z{t+6}.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        negative_samples=16,
        k_predictions=5,
        weight_init=True,
    ):
        super().__init__()
        self.negative_samples = negative_samples
        self.k_predictions = k_predictions

        # 1x1 convolutions to sample from the channel dimension
        # representations converted from in_channels to out_channels dimensions
        self.W_k = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(self.k_predictions)
        )

        if weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                if m in self.W_k:
                    # nn.init.kaiming_normal_(
                    #     m.weight, mode="fan_in", nonlinearity="tanh"
                    # )
                    makeDeltaOrthogonal(m.weight, nn.init.calculate_gain("sigmoid"))

    def forward(self, z, c, skip_step=1):
        """
        :param z: positive samples z_{t} (used to create z_{t+k} for each k)
        :param c: context (reference samples). In audio experiments, this is an
                  autoregressive representation which captures the entire temporal
                  history up until time t. In vision experiments, this is simply
                  z_{t}, which is the same as parameter z.
        :param skip_step: Number of steps to skip between the reference sample and
                          the positive samples. Defaults to 1, as in the paper.
                          Skipping 1 step prevents overlapping samples from being
                          compared with the default overlap. This parameter should
                          probably be increased for larger overlap values.
        """
        batch_size = z.shape[0]

        # For each k in k_predictions, store a set of log_fk and true_f values
        log_f_list, true_f_list = [], []
        for k in range(1, self.k_predictions + 1):
            # Compute log f(c_t, x_{t+k}) = z^T_{t+k} W_k c_t

            # First half of bilinear model, compute z^T_{t+k} W_k:
            ztwk = (
                self.W_k[k - 1]  # 1x1 convolution
                .forward(z[:, :, (k + skip_step) :, :])  # Bx, C , H , W
                .permute(2, 3, 0, 1)  # H, W, Bx, C
                .contiguous()
            )  # y, x, b, c

            # Take random samples from z_{t+k} W_k as contrastive samples
            ztwk_shuf = ztwk.view(
                ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3]
            )  # y * x * batch, c
            rand_index = torch.randint(
                ztwk_shuf.shape[0],  # y *  x * batch
                (ztwk_shuf.shape[0] * self.negative_samples, 1),
                dtype=torch.long,
                device=ztwk_shuf.device,
            )
            rand_index = rand_index.repeat(1, ztwk_shuf.shape[1])
            ztwk_shuf = torch.gather(
                ztwk_shuf, dim=0, index=rand_index, out=None
            )  # y * x * b * n, c
            ztwk_shuf = ztwk_shuf.view(
                ztwk.shape[0],
                ztwk.shape[1],
                ztwk.shape[2],
                self.negative_samples,
                ztwk.shape[3],
            ).permute(
                0, 1, 2, 4, 3
            )  # y, x, b, c, n

            # Multiply ztwk and context for full bilinear model
            context = (
                c[:, :, : -(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2)
            )  # y, x, b, 1, c

            log_fk_main = torch.matmul(context, ztwk.unsqueeze(-1)).squeeze(
                -2
            )  # y, x, b, 1

            log_fk_shuf = torch.matmul(context, ztwk_shuf).squeeze(-2)  # y, x, b, n

            # Stack negative samples below positive samples
            log_fk = torch.cat((log_fk_main, log_fk_shuf), 3)  # y, x, b, 1+n
            log_fk = log_fk.permute(2, 3, 0, 1)  # b, 1+n, y, x

            # Positive samples are at index 0
            true_fk = torch.zeros(
                (batch_size, log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=log_fk.device,
            )  # b, y, x

            # #append results to list
            log_f_list.append(log_fk)
            true_f_list.append(true_fk)

        return log_f_list, true_f_list
