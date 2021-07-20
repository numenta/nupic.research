# Finetuning workflow documentation

This document is for outlining the process of finetuning pretrained models. This may involve several pretrained models to experiment with. Each pretrained model might have different hyperparameters for each of 9 tasks to finetune on, and multiple runs with select hyperparameters for each task. This document describes the current workflow for dealing with this.

## Finetuning a pretrained model

Use the typical `python run.py <experiment>` to run a finetuning experiment. Finetuning configs are not all located in the experiments/finetuning.py file. For instance, trifecta models have finetuning configs in the experiments/trifecta.py file. See finetuning_bert700k_glue for an example of setting up a finetuning config. Here is a checklist of things to consider before kicking off a finetuning experiment. 

- Make sure that the correct model type and path are specified. For instance, sparse models should generally have "fully_static_sparse_bert" as the model type. The model_name_or_path should point to where a pretrained model was saved.
- Make sure that max_steps and eval_steps have been set appropriately. If max_steps is too large, then unless EarlyStoppingCallback is enabled, training could take a very long time. Making eval_steps too small can also slow down training, especially for large datasets like mnli. Definitely make sure eval_steps is less than max_steps. Generally it is recommended that max_steps be a multiple of eval_steps.
- Unless you have a specific reason not to, make sure "load_best_model_at_end" is on. The rest of the pipeline assumes this is the case. This will automatically load the model with the best "metric_for_best_model" at the end of training. It also saves a model every time a model is evaluated, specified by eval_steps.
- Makre sure "metric_for_best_model" is specified, otherwise you'll get an error when "load_best_model_at_end" is on. Make sure the metric pertains to the task at hand. For instance, "eval_accuracy" is not measured in COLA. See finetuning_constants.py for information on which tasks use which metrics, and other information like dataset size. 
- Make sure the correct callbacks are in play. For sparse models, you must have RezeroWeightsCallback (or you'll get an error). For finetuning experiments, please ensure TrackEvalMetrics callback is in place, as this is used to track evaluation metrics throughout training and subsequent analysis relies on it.

For additional information, please see [huggingface documentation](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments).

## Syncing and exporting finetuning results

## Hyperparameter search

## Syncing and exporting hyperparameter search results

## Finetuning with new hyperparameters

## Uploading to the glue leaderboard