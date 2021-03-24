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
import functools
import os
from collections import namedtuple
from hashlib import blake2b

import torch
import torch.nn.functional as F
import transformers
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, BertConfig


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


def soft_cross_entropy_masked(output, target, padding_mask):
    """
    Soft cross entropy which accepts a padding mask.
    Padding mask ensure loss is computed only for the tokens which are being predicted
    in masked language modeling.
    """
    loss_vec = -target * F.log_softmax(output, dim=-1)
    loss_vec.masked_fill_(padding_mask, 0.0)
    loss = loss_vec.sum()
    num_active_elements = padding_mask.numel() - padding_mask.sum()
    return loss / num_active_elements


def get_logits(output):
    return output["logits"] if isinstance(output, dict) else output[0]


def calculate_kd_loss(batch, student, teacher):
    kd_temperature = 1.0
    w_factor = 1.0
    loss_fn = soft_cross_entropy_masked

    # calculate student softmax
    student_outputs = student(**batch)
    student_logits = get_logits(student_outputs)

    # calculate teacher logits
    with torch.no_grad():
        teacher_outputs = teacher(**batch)
        teacher_logits = get_logits(teacher_outputs)
        kd_target = F.softmax(teacher_logits / kd_temperature, dim=-1) * w_factor

    # get and apply padding mask
    labels = batch["labels"]
    if labels.dim() == kd_target.dim() - 1:
        labels = labels.unsqueeze(-1)
    padding_mask = labels.eq(-100)

    # calculate raw loss
    loss = loss_fn(output=student_logits, target=kd_target, padding_mask=padding_mask)
    return loss, student_outputs, teacher_outputs


def calculate_smoothed_loss(batch, model):
    ignore_index = -100
    epsilon = 0.1

    labels = batch["labels"]
    model_output = model(**batch)
    logits = (
        model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    )
    log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0.
    # The padding_mask will ignore them in any case.
    labels.clamp_min_(0)
    nll_loss = log_probs.gather(dim=-1, index=labels)
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active
    # elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


def prepare_batch(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    return batch


def calculate_batch_acc(batch, outputs):
    logits = outputs.logits[:, :, :].squeeze()
    labels = batch["labels"].squeeze()
    valid_entries = labels != -100
    total_predicted = torch.sum(valid_entries).item()
    total_correct = torch.sum(
        torch.argmax(logits[valid_entries], dim=1) == labels[valid_entries]
    ).item()

    return total_correct / total_predicted


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


if __name__ == "__main__":

    # ----- load datasets, tokenizer, data collator and data loader
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

    # load dataloader
    dataset = tokenized_datasets["train"]
    dataloader = DataLoader(
        dataset, batch_size=32, collate_fn=data_collator, sampler=RandomSampler(dataset)
    )
    print("Data utilities loaded.")

    # ----- load models
    # load student model
    device = torch.device("cuda")
    config_kwargs = dict(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=128,
        intermediate_size=512,
        vocab_size=28996,  # distilling from a bert_cased, resize_token_embedding
        max_position_embeddings=128,
    )
    student_config = BertConfig(**config_kwargs)
    student_model = AutoModelForMaskedLM.from_config(student_config)
    student_model = student_model.to(device)

    # load a teacher model - bert base
    model_name_or_path = (
        "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
    )
    teacher_config = AutoConfig.from_pretrained(
        model_name_or_path, revision="main", use_auth_token=False, local_files_only=True
    )
    teacher_model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path, config=teacher_config, use_auth_token=False
    )
    teacher_model = resize_position_embeddings(teacher_model, new_seq_length=128)
    teacher_model = teacher_model.to(device)
    print("Models loaded.")

    def get_loss(outputs):
        return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    # ----- train
    epochs = 10
    optim = torch.optim.Adam(lr=1e-3, params=student_model.parameters())
    for epoch in range(epochs):
        print(f"------ Starting epoch {epoch}")
        for idx, batch in enumerate(dataloader):

            batch = prepare_batch(batch, device)

            # compute loss and accs
            kd_loss, student_outputs, teacher_outputs = calculate_kd_loss(
                batch, student=student_model, teacher=teacher_model
            )
            if idx % 50 == 0:
                # student metrics
                target_loss = get_loss(student_outputs)
                target_acc = calculate_batch_acc(batch, student_outputs)
                # teacher metrics, baseline
                teacher_loss = get_loss(teacher_outputs)
                teacher_acc = calculate_batch_acc(batch, teacher_outputs)
                # report
                print(
                    f"kd loss: {kd_loss.item():.4f}, "
                    f"tgt loss: {target_loss.item():.4f}, "
                    f"tgt acc: {target_acc:.2f}, "
                    f"tch loss: {teacher_loss.item():.4f} ,"
                    f"tch acc: {teacher_acc:.4f}"
                )

            # learn
            kd_loss.backward()
            optim.step()
            optim.zero_grad()
