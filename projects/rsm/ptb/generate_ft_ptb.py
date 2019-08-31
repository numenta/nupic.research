from torchnlp.datasets import penn_treebank_dataset
import fasttext
import os
import sys

PATH = "/home/ubuntu"
# PATH = "/Users/jgordon"

print("Maybe download ptb...")
penn_treebank_dataset(PATH + "/nta/datasets/PTB", train=True, test=True)


PTB_TRAIN_PATH = PATH + "/nta/datasets/PTB/ptb.train.txt"

if len(sys.argv) > 1:
	epoch = int(sys.argv[1])
else:
	epoch = 5

model = fasttext.train_unsupervised(PTB_TRAIN_PATH, model='skipgram', minCount=1, epoch=epoch)
embed_dir = PATH + "/nta/datasets/embeddings"
filename = PATH + "/nta/datasets/embeddings/ptb_fasttext_e%d.bin" % epoch
if not os.path.exists(embed_dir):
    os.makedirs(embed_dir)

print("Saved %s" % filename)
model.save_model(filename)