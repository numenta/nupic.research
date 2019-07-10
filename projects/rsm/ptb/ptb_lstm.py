import torch.nn as nn


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_in=1, d_out=1, vocab_size=10000, embed_dim=200, nhid=650, dropout=0.5, nlayers=2, tie_weights=False):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        rnn_input_dim = d_in
        self.encoder = None
        if embed_dim:
            self.encoder = nn.Embedding(d_in, embed_dim)
            rnn_input_dim = embed_dim

        self.rnn = nn.LSTM(rnn_input_dim, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, d_out)

        if tie_weights:
            if nhid != embed_dim:
                raise ValueError('When using the tied flag, nhid must be equal to embed_dim')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        if self.encoder:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # Takes input of shape (seq_len, batch, input_size)
        input = input.view(1, input.size(0), input.size(1))
        if self.encoder:
            emb = self.drop(self.encoder(input))
        else:
            emb = input
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    def _track_weights(self):
        return {}

    def _register_hooks(self):
        pass