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

"""Dataloading functions"""

import functools
import os
from collections import namedtuple
from hashlib import blake2b

import torch
import transformers
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import AutoTokenizer


def get_dataset_wkbc():
    """Returns a dataloader with customized 1% wikipedia + 8% book corpus dataset"""

    # define data args
    data_args = dict(
        max_seq_length=128,
        dataset_name=("wikipedia_plus_bookcorpus",),
        dataset_config_name=(None,),
        tokenized_data_cache_dir="/mnt/efs/results/preprocessed-datasets/text",
        pad_to_max_length=False,
        data_collator="DataCollatorForWholeWordMask",
        mlm_probability=0.15,
    )
    data_args = namedtuple("data_args", data_args)(**data_args)

    # load tokenized dataset
    dataset_folder = hash_dataset_folder_name(data_args)
    dataset_path = os.path.join(
        os.path.abspath(data_args.tokenized_data_cache_dir), str(dataset_folder)
    )
    tokenized_datasets = load_from_disk(dataset_path)

    # load tokenizer
    tokenizer_kwargs = dict(
        revision="main",
        cache_dir="/mnt/efs/results/pretrained-models/huggingface",
        use_fast=True,
        use_auth_token=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", **tokenizer_kwargs)

    # load data collator
    data_collator = getattr(transformers, data_args.data_collator)(
        tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
    )

    print("Dataset loaded.")

    DataAttributes = namedtuple(
        "DataAttributes",
        ["train_dataset", "eval_dataset", "data_collator", "tokenizer"]
    )
    data_attributes = DataAttributes(
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        data_collator,
        tokenizer,
    )

    return data_attributes


def get_dataloader_wkbc(batch_size=32, load_validation_dataloader=False):
    """Returns a dataloader with customized 1% wikipedia + 8% book corpus dataset"""

    data_attrs = get_dataset_wkbc()

    train_dataloader = DataLoader(
        data_attrs.train_dataset,
        batch_size=batch_size,
        collate_fn=data_attrs.data_collator,
        sampler=RandomSampler(data_attrs.train_dataset)
    )
    if load_validation_dataloader:
        eval_dataloader = DataLoader(
            data_attrs.eval_dataset,
            batch_size=batch_size,
            collate_fn=data_attrs.data_collator,
        )
        return train_dataloader, eval_dataloader
    else:
        return train_dataloader


def hash_dataset_folder_name(data_args):
    """
    Creates a hashed name for the dataset folder comprised of the dataset_name and
    dataset_name_config. As well, the following data_args are included unless their
    default values are used.
        - max_seq_length (default None)

    More arguments can be added to the hashed name as needed.
    """
    defaults = dict(
        max_seq_length=None,
    )

    dataset_folder = "-".join([
        f"{name}_{config}" for name, config in
        zip(data_args.dataset_name, data_args.dataset_config_name)
    ])

    for arg, default in defaults.items():
        if getattr(data_args, arg) != default:
            non_default = getattr(data_args, arg)
            dataset_folder += f" ({arg}={non_default})"

    hashed_folder_name = blake2b(dataset_folder.encode(), digest_size=20).hexdigest()
    print(f"Hashing dataset folder name '{dataset_folder}' to '{hashed_folder_name}'")
    return hashed_folder_name


def resize_position_embeddings(model, new_seq_length):
    """
    Resizes model's position embeddings matrices if the size of max position embedding
    doesn't match new sequence length.
    (size of position embedding equals size of the attention window)

    :param :new_seq_length Tokenizer sequence length.
    """

    # Functions to recursively get and set attributes. Source:
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties  # noqa: E501
    def rsetattr(obj, attr, val):
        pre, _, post = attr.rpartition(".")
        return setattr(rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))

    # Find and replace all position embeddings if the size of position embedding
    # doesn't match new sequence length
    for module_name, module in model.named_modules():
        if "position_embeddings" in module_name:
            original_embed_data = module.weight.data
            max_position_embeddings, embed_hidden_size = original_embed_data.size()
            if max_position_embeddings != new_seq_length:
                new_embed = torch.nn.Embedding(new_seq_length, embed_hidden_size)
                new_embed.weight.data[:, :] = original_embed_data[:new_seq_length, :]
                rsetattr(model, module_name, new_embed)

    return model
