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
import sys

import fasttext
from torchnlp.datasets import penn_treebank_dataset

PATH = "/home/ubuntu"
# PATH = "/Users/jgordon"

print("Maybe download ptb...")
penn_treebank_dataset(PATH + "/nta/datasets/PTB", train=True, test=True)


PTB_TRAIN_PATH = PATH + "/nta/datasets/PTB/ptb.train.txt"

if len(sys.argv) > 1:
    epoch = int(sys.argv[1])
else:
    epoch = 5

model = fasttext.train_unsupervised(
    PTB_TRAIN_PATH, model="skipgram", minCount=1, epoch=epoch
)
embed_dir = PATH + "/nta/datasets/embeddings"
filename = PATH + "/nta/datasets/embeddings/ptb_fasttext_e%d.bin" % epoch
if not os.path.exists(embed_dir):
    os.makedirs(embed_dir)

print("Saved %s" % filename)
model.save_model(filename)
