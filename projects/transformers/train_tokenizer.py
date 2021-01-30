#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
#

"""
Alternative ways to load tokenizer from vocab.txt or train it from dataset
using tokenizers lib
"""

import glob
import os

from tokenizers import (
    BertWordPieceTokenizer,
)

def load_pretrained_bert_tokenizer(vocab_file=None):
    """Create tokenizer from file, using Transformers library"""

    from transformers import BertTokenizerFast

    if vocab_file is None:
        vocab_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bert-base-uncased-vocab.txt"
        )

    tokenizer = BertTokenizerFast(
        vocab_file=vocab_file,
        # following arguments are all same as default, listed for clarity
        clean_text=True,
        tokenize_chinese_chars=True,
        do_lower_case=True,
        strip_accents=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    return tokenizer


def load_from_files_bert_tokenizer(path_to_files=None, vocab_size=30000):
    """
    Adapted from https://github.com/huggingface/tokenizers/tree/master/bindings/python/examples
    If used frequently, save the model to avoid reloading (see example above)
    """

    if path_to_files is None:
        path_to_files = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sample_files"
        )
    # parse more complex patterns if used
    files = glob.glob(path_to_files)

    # Create tokenizer using tokenizer
    tokenizer = BertWordPieceTokenizer(
        strip_accents=True,
        # following arguments are all same as default, listed for clarity
        clean_text=True,
        handle_chinese_chars=True,
        lowercase=True,
    )

    # And finally train
    tokenizer.train(
        files,
        # following arguments are all same as default, listed for clarity
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    return tokenizer


def load_from_dataset_bert_tokenizer(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    vocab_size=30000
):
    """
    Adapted from https://github.com/huggingface/tokenizers/tree/master/bindings/python/examples
    If used frequently, save the model to avoid reloading

    tokenizer 0.10.0 required to train from dataset, but not supported by stable version
    of hugging face or datasets yet
    """

    from datasets import load_dataset

    tokenizer = BertWordPieceTokenizer(
        strip_accents=True,
        # following arguments are all same as default, listed for clarity
        clean_text=True,
        handle_chinese_chars=True,
        lowercase=True,
    )

    dataset = load_dataset(dataset_name, dataset_config_name)

    # Build an iterator over this dataset
    def batch_iterator():
        batch_length = 1000
        for i in range(0, len(dataset["train"]), batch_length):
            yield dataset["train"][i : i + batch_length]["text"]

    # Train
    tokenizer.train_from_iterator(
        batch_iterator(),
        length=len(dataset["train"]),
        # following arguments are all same as default, listed for clarity
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    return tokenizer


# if __name__ == "__main__":
#     tokenizer = load_pretrained_bert_tokenizer()
#     tokenizer = load_from_files_bert_tokenizer(vocab_size=100)
#     tokenizer = load_from_dataset_bert_tokenizer()
#     print(tokenizer)
