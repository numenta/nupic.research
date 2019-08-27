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

import torch.nn as nn


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        d_in=1,
        d_out=1,
        vocab_size=10000,
        embed_dim=200,
        nhid=650,
        dropout=0.0,
        nlayers=1,
        tie_weights=False,
    ):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        rnn_input_dim = d_in

        self.encoder = None
        self.rnn = nn.LSTM(rnn_input_dim, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, d_out)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        if self.encoder:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        # Takes input x of shape (batch, input_size)
        x = x.view(1, x.size(0), x.size(1))
        if self.encoder:
            emb = self.drop(self.encoder(x))
        else:
            emb = x
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded.view(output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),  # h_0
            weight.new_zeros(self.nlayers, bsz, self.nhid)  # c_0
        )

    def _track_weights(self):
        return {}

    def _post_epoch(self, epoch):
        pass

    def _register_hooks(self):
        pass


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        d_in=1,
        d_out=1,
        vocab_size=10000,
        embed_dim=200,
        nhid=650,
        dropout=0.0,
        nlayers=2,
        tie_weights=False,
    ):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        rnn_input_dim = d_in

        self.encoder = None
        self.rnn = nn.RNN(rnn_input_dim, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, d_out)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        if self.encoder:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        # Takes input x of shape (batch, input_size)
        x = x.view(1, x.size(0), x.size(1))
        if self.encoder:
            emb = self.drop(self.encoder(x))
        else:
            emb = x
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded.view(output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def _track_weights(self):
        return {}

    def _post_epoch(self, epoch):
        pass

    def _register_hooks(self):
        pass
