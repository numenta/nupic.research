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

"""Training and eval functions"""

import math
import os

from transformers.integrations import TensorBoardCallback

import ray
from ray import tune

__all__ = [
    "run_hf",
    "run_ray_single_instance",
    "run_ray_distributed"
]


def run_hf(trainer, logger, output_dir, save_model=True, evaluate=True):
    """Use only hugging face to train and evaluate. Default"""

    # Train model to given number of epochs
    train_result = trainer.train(resume_from_checkpoint=None)

    if save_model:
        # Saves model and tokenizer
        trainer.save_model()

        # Save results
        output_train_file = os.path.join(output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, Trainer.save_model saves only tokenizer and model
            trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    if evaluate:
        results = {}
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(output_dir, "eval_results_mlm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        logger.info(results)


# ----- Ray Distriuted Training -----------


def run_ray_single_instance(trainer, logger, **kwargs):
    """Run with ray in a single instance. Tested."""

    # adapted from HF integrations
    prefix_checkpoint_dir = "checkpoint"

    def _objective(trial, checkpoint_dir=None):
        model_path = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(prefix_checkpoint_dir):
                    model_path = os.path.join(checkpoint_dir, subdir)

        trainer.objective = None
        trainer.train(model_path=model_path, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            # Logs at the end of the objective function, useful for hp search only
            trainer._tune_save_checkpoint()
            # Q: what else is reporting, when not reporting hp search metrics?
            tune.report(**metrics, done=True)

    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)

    # Setup default `resources_per_trial` and `reporter`.
    if "resources_per_trial" not in kwargs and trainer.args.n_gpu > 0:
        # `args.n_gpu` is considered the total number of GPUs that will be split
        # among the `n_jobs`
        n_jobs = int(kwargs.pop("n_jobs", 1))
        num_gpus_per_trial = trainer.args.n_gpu
        if num_gpus_per_trial / n_jobs >= 1:
            num_gpus_per_trial = int(math.ceil(num_gpus_per_trial / n_jobs))
        kwargs["resources_per_trial"] = {"gpu": num_gpus_per_trial}

    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])

    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        # `keep_checkpoints_num=0` would disabled checkpointing
        trainer.use_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                "Currently keeping {} checkpoints for each trial."
                "Checkpoints are large, consider setting `keep_checkpoints_num=1`."
            )

    # run tune
    tune.run(_objective, **kwargs)


def run_ray_distributed(trainer, logger, **kwargs):
    """Run with ray in a multiple instances. Not tested."""

    # Connect to ray
    ray.init(
        address="auto",
        _redis_password="a1b2c3",
        local_mode=False,
        include_dashboard=True, dashboard_host="0.0.0.0",
    )

    # adapted from HF integrations
    prefix_checkpoint_dir = "checkpoint"

    def _objective(trial, checkpoint_dir=None):
        model_path = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(prefix_checkpoint_dir):
                    model_path = os.path.join(checkpoint_dir, subdir)

        trainer.objective = None
        trainer.train(model_path=model_path, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            # Logs at the end of the objective function, useful for hp search only
            trainer._tune_save_checkpoint()
            # Q: what else is reporting, when not reporting hp search metrics?
            tune.report(**metrics, done=True)

    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)

    # Setup default `resources_per_trial` and `reporter`.
    if "resources_per_trial" not in kwargs and trainer.args.n_gpu > 0:
        # `args.n_gpu` is considered the total number of GPUs that will be split
        # among the `n_jobs`
        n_jobs = int(kwargs.pop("n_jobs", 1))
        num_gpus_per_trial = trainer.args.n_gpu
        if num_gpus_per_trial / n_jobs >= 1:
            num_gpus_per_trial = int(math.ceil(num_gpus_per_trial / n_jobs))
        kwargs["resources_per_trial"] = {"gpu": num_gpus_per_trial}

    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])

    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        # `keep_checkpoints_num=0` would disabled checkpointing
        trainer.use_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                "Currently keeping {} checkpoints for each trial."
                "Checkpoints are large, consider setting `keep_checkpoints_num=1`."
            )

    # run tune
    tune.run(_objective, **kwargs)

    # shutdown Ray after training is complete
    ray.shutdown()
