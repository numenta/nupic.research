# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from nupic.research.frameworks.mandp.foliage import FoliageDataset


class MandpAutoencoder(pl.LightningModule):
    def __init__(self, num_modules=30, viewport_height=64, viewport_width=64,
                 batch_size=64, transpose_decoder=True, step_size=200,
                 krecon=20.0, kmag=40.0, kphase=25.0, local_delta=False):
        super().__init__()
        self.viewport_height = viewport_height
        self.viewport_width = viewport_width
        self.batch_size = batch_size
        self.step_size = step_size

        in_features = viewport_height * viewport_width
        self.encoder = nn.Linear(in_features, num_modules * 2, bias=False)

        if transpose_decoder:
            self.decoder = None
        else:
            self.decoder = nn.Linear(num_modules * 2, in_features, bias=False)

        self.krecon = krecon
        self.kmag = kmag
        self.kphase = kphase
        self.local_delta = local_delta

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # Batch contains a set of trajectories
        encodings = self(batch)
        if self.decoder is not None:
            decodings = self.decoder(encodings)
        else:
            decodings = torch.nn.functional.linear(
                encodings, self.encoder.weight.transpose(0, 1))

        reconstruction_loss = self.krecon * F.mse_loss(decodings, batch,
                                                       reduction="mean")

        complex_encodings = torch.view_as_complex(
            encodings.view(encodings.shape[0], encodings.shape[1],
                           -1, 2))

        # Minimize changes in magnitude (A perfect network will not vary the
        # magnitude at all)
        mags = complex_encodings.abs()
        if self.local_delta:
            delta_mags = mags[:, 1:, :] - mags[:, :-1, :]
            constant_mag_loss = self.kmag * delta_mags.square().mean()
        else:
            constant_mag_loss = self.kmag * torch.var(mags, dim=1).mean()

        # Minimize changes in phase velocity (A perfect network will use
        # constant changes in phase)
        phases = torch.atan2(torch.imag(complex_encodings),
                             torch.real(complex_encodings))
        # phases = complex_encodings.angle()   # Not supported by autograd...
        delta_phases = phases[:, 1:, :] - phases[:, :-1, :]
        delta_phases = torch.where(delta_phases <= -math.pi,
                                   delta_phases + (2 * math.pi),
                                   delta_phases)
        delta_phases = torch.where(delta_phases > math.pi,
                                   delta_phases - (2 * math.pi),
                                   delta_phases)
        if self.local_delta:
            d2_phases = delta_phases[:, 1:] - delta_phases[:, :-1]
            linear_phase_loss = self.kphase * d2_phases.square().mean()
        else:
            linear_phase_loss = self.kphase * torch.var(delta_phases, dim=1).mean()

        return (reconstruction_loss
                + constant_mag_loss
                + linear_phase_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=0.5)
        lr_scheduler = dict(scheduler=lr_scheduler, interval="step")

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        dataset = FoliageDataset()
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                           num_workers=8, pin_memory=True)
