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

from ray import tune
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    BertConfig,
    Trainer,
    TrainingArguments,
)

from dataloading_utils import get_dataset_wkbc, resize_position_embeddings

if __name__ == "__main__":

    # ------- Prepare Data
    data_attrs = get_dataset_wkbc()
    print(f"train dataset size: {len(data_attrs.train_dataset):,}", )
    print(f"eval dataset size: {len(data_attrs.eval_dataset):,}")

    # ------- Prepare Model

    def model_init():
        config_kwargs = dict(
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=128,
            intermediate_size=512,
            vocab_size=28996,
            max_position_embeddings=128,
        )
        config = BertConfig(**config_kwargs)
        return AutoModelForMaskedLM.from_config(config)

    def model_init_pretrained():
        # load a a
        model_name_or_path = (
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        )
        teacher_config = AutoConfig.from_pretrained(
            model_name_or_path, revision="main", use_auth_token=False,
            local_files_only=True
        )
        teacher_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=teacher_config, use_auth_token=False
        )
        teacher_model = resize_position_embeddings(teacher_model, new_seq_length=128)
        return teacher_model

    # def compute_metrics(ep):
    #     """
    #     The function that will be used to compute metrics at evaluation. Must take a
    #     :class:`~transformers.EvalPrediction` and return a dictionary string to metric
    #     values.
    #     """
    #     return {"accuracy": (ep.preds == ep.labels).astype(np.float32).mean().item()}

    # Evaluate during training and a bit more often than the default to be able to
    # prune bad trials early. Disabling tqdm is a matter of preference.
    training_args = TrainingArguments(
        eval_accumulation_steps=None,  # required?
        dataloader_drop_last=True,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=20,
        max_steps=100,
        do_train=True,
        do_eval=False,
        do_predict=False,
        output_dir="./",
        disable_tqdm=True,
        load_best_model_at_end=True,  # will assign to self.model,
        # metric_for_best_model="accuracy",
        # evaluate_during_training=True  # not available in recent
    )

    # Create Trainer
    trainer = Trainer(
        args=training_args,
        model_init=model_init_pretrained,
        # compute_metrics=compute_metrics,
        tokenizer=data_attrs.tokenizer,
        train_dataset=data_attrs.train_dataset,
        eval_dataset=data_attrs.eval_dataset.shard(index=1, num_shards=100),
        data_collator=data_attrs.data_collator,
    )

    # ------- Optimization

    def hp_space(trial):
        return dict(
            learning_rate=tune.loguniform(1e-4, 1e-2),
        )

    # def compute_objective(metrics):
    #     """
    #     The default objective to maximize/minimize when doing an hyperparameter search
    #     It is the evaluation loss if no metrics are provided to the
    #     :class:`~transformers.Trainer`, the sum of all metrics otherwise.

    #     Args:
    #         metrics (:obj:`Dict[str, float]`): The metrics returned by evaluate method

    #     Return:
    #         :obj:`float`: The objective to minimize or maximize
    #     """
    #     return metrics["accuracy"]

    def custom_compute_objective(metrics):
        return metrics["eval_loss"]

    # this is opposed to run
    # if I do hp search, this goes in place of the best evaluation

    # use sharding to evaluate faster

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        n_trials=3,  # number of hyperparameter samples
        hp_space=hp_space,
        compute_objective=custom_compute_objective,
        # other parameters:
        # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
        # search_alg=ray.tune.suggest.hyperopt.HyperOptSearch(),
        # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
        # scheduler=AsyncHyperBand(),  # aggressive termination of trials
        # n_jobs=4,  # number of parallel jobs, if multiple GPUs
    )

    print(best_trial)
