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

from __future__ import print_function

import math
import os
from io import open

import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.EOS_TOKEN = "</s>"  # "<eos>"
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "ptb.train.txt"))
        self.valid = self.tokenize(os.path.join(path, "ptb.valid.txt"))
        self.test = self.tokenize(os.path.join(path, "ptb.test.txt"))

    def encode(self, words):
        return [self.dictionary.word2idx[word] for word in words.split()]

    def read_out(self, ids):
        out = []
        if isinstance(ids, int):
            ids = [ids]
        for _id in ids:
            out.append(self.dictionary.idx2word[_id])
        return " ".join(out)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + [self.EOS_TOKEN]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            ids = torch.zeros(tokens).long()
            token = 0
            for line in f:
                words = line.split() + [self.EOS_TOKEN]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids


class BitwiseWordEmbedding(object):
    def __init__(self, vocab_size=10000, dim=28):
        self.vocab_size = vocab_size
        self.embedding_dict = {}
        self.dim = dim
        self.generate_embeddings()

    def generate_embeddings(self):
        for i in range(self.vocab_size):
            self.embedding_dict[i] = self.embed(i)

    def embed(self, i):
        first = "{0:b}".format(i).zfill(self.dim // 2)
        return self.vectorize(first + self.inverse(first))

    def inverse(self, binstr):
        return "".join("1" if x == "0" else "0" for x in binstr)

    def vectorize(self, s):
        return torch.FloatTensor([1 if x == "1" else 0 for x in s])


def perpl(nll):
    """Perplexity is exp(negative_log_likelihood). Clips very large values."""
    # Avoid overflow
    return math.exp(nll) if nll < 50 else 1000000
