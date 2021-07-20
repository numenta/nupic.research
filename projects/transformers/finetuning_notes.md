# Finetuning workflow documentation

This document is for outlining the process of finetuning pretrained models. This may involve several pretrained models to experiment with. Each pretrained model might have different hyperparameters for each of 9 tasks to finetune on, and multiple runs with select hyperparameters for each task. This document describes the current workflow for dealing with this.

## Finetuning a pretrained model

Use the typical `python run.py <experiment>` to run a finetuning experiment. Finetuning configs are not all located in the experiments/finetuning.py file. For instance, trifecta models have finetuning configs in the experiments/trifecta.py file. See finetuning_bert700k_glue for an example of setting up a finetuning config. Here is a checklist of things to consider before kicking off a finetuning experiment. 

- Make sure that the correct model type and path are specified. For instance, sparse models should generally have "fully_static_sparse_bert" as the model type. The model_name_or_path should point to where a pretrained model was saved.
- Make sure that max_steps and eval_steps have been set appropriately. If max_steps is too large, then unless EarlyStoppingCallback is enabled, training could take a very long time. Making eval_steps too small can also slow down training, especially for large datasets like mnli. Definitely make sure eval_steps is less than max_steps. Generally it is recommended that max_steps be a multiple of eval_steps.
- Unless you have a specific reason not to, make sure "load_best_model_at_end" is on. The rest of the pipeline assumes this is the case. This will automatically load the model with the best "metric_for_best_model" at the end of training. It also saves a model every time a model is evaluated, specified by eval_steps.
- Makre sure "metric_for_best_model" is specified, otherwise you'll get an error when "load_best_model_at_end" is on. Make sure the metric pertains to the task at hand. For instance, "eval_accuracy" is not measured in COLA. See finetuning_constants.py for information on which tasks use which metrics, and other information like dataset size. 
- Make sure the correct callbacks are in play. For sparse models, you must have RezeroWeightsCallback (or you'll get an error). For finetuning experiments, please ensure TrackEvalMetrics callback is in place, as this is used to track evaluation metrics throughout training and subsequent analysis relies on it.
- If you want to potentially upload predictions from this experiment to the GLUE leaderboard, ensure do_predict is on.

For additional information, please see [huggingface documentation](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments).

Note that when you run finetuning, some tasks will go for multiple runs. As of recently, models and predictions will be stored separately for each run. You need to know which run did best for each task, to determine which model to use to upload predictions to the glue leaderboard. Our code currently stores predictions for all runs, and then finds the model with the best eval scores, and then creates a symlink {task_name}_best.tsv that refers to the {task_name}.tsv predictions file for that model

## Syncing and exporting finetuning results

In ./scripts/finetuning_sync_functions.sh are a series of shell functions that can be adapted for syncing finetuning results. Currently, I keep these in my .zshrc file. Sometimes, I find out there was a bug in a run, and I manually use ray rsync and point to a task_results file, and sync to a local file with a name that indicates the type of bug. I keep a local list of configs (one config per line, no trailing whitespace, one empty line at the end) that the above functions rely on to sync. Example usage:

`sync_finetuning ~/nta/ray_config/my_ray_config.yaml ~/nta/finetuning/configs.txt`

After syncing results to my local machine, I use export_finetuning_results.py. Since there are multiple models to keep track of simultaneously, sync_finetuning syncs all specified models, and scripts/analyze_all_ft.sh saves the results to results/finetuning_results.{csv,md}. This file just contains calls to export_finetuning_results.py, so update according to your needs.

## Hyperparameter search

All hyperparameter tuning experiment configs are currently contained in experiments/hpsearch. To run hyperparameter tuning for each of the 9 GLUE tasks, follow the examples of hp_search_finetuning_trifecta_85_100k_small_tasks ro hp_search_finetuning_trifecta_85_100k_big_tasks. 

Hyperparameter tuning takes a long time because it involves training models to completion on each task multiple times. A hack to speed this process up is to run multiple finetuning experiments simultaneously on different clusters. You can kill either one at any time. This has the benefit of running in parallel, but by default, ray tune saves results in such a way that hyerparamter configs and evalutaion scores won't be overwritten. You can run multiple finetuning experiments and then use export_finetuning_hp_search_results.py to get the best hyperparameters.

## Syncing and exporting hyperparameter search results

To sync result, adapt the functions in scripts/finetuning_sync_functions.sh. Example:

`sync_hp_search ~/nta/ray_config/my_ray_config.yaml ~/nta/hp_search/configs.txt`

Again, its a hassle to run export_finetuning_hp_search_results.py multiple times to export results for each pretrained model. So, there's analyze_all_hp_search.sh which calls the python script once for each hyperparameter search experiment. You can run both at once by adapting the sync_and_export_hp function.

export_finetuning_hp_search_results.py will take the best hyperparameters seen so far, for each task, for each pretreained model, and save them to experiments/hp_finetuning/some_model/task_name_hps.p. There is aditional code in this python script for making plots and so on.

## Finetuning with new hyperparameters

Create a config in hpchase, which runs finetuning again. The parent config should be a config for finetuning the same pretrained model, and now you will just update it with hyperparameters per task. You can do this by making a call to update_task_hyperparams. Then you can use the usual 

`python run.py <my_hyperparam_tuned_config>`

## Uploading to the glue leaderboard

To sync the results of the above, adapt the code (not yet tested) from the sync_finetuning_test function found in scripts/finetuning_sync_functions.sh. Recall that a symlink is created for each finetuning task that points to the predictions on the test set for the model with the best eval score. This will sync that file (for each task) to a local file without the _best suffix.

Once the prediction files are all in a single directory pertaining to a single pretrained model, then run zip_glue to compress the predictions and upload them to the [glue leaderboard](https://gluebenchmark.com/leaderboard). 