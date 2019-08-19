from torchnlp.datasets import penn_treebank_dataset
import fasttext

print("Maybe download ptb...")
penn_treebank_dataset("/home/ubuntu/nta/datasets/PTB", train=True, test=True)

PTB_TRAIN_PATH = "/home/ubuntu/nta/datasets/PTB/ptb.train.txt"

model = fasttext.train_unsupervised(PTB_TRAIN_PATH, model='skipgram')
filename = "/home/ubuntu/nta/datasets/embeddings/ptb_fasttext.bin"
print("Saved %s" % filename)
model.save_model(filename)