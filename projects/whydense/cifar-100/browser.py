"""
  Module for browsing and manipulating experiment results directories created
  by Ray Tune.

  As it is will just process and return dataframe, it won't keep internal state, so converted to a module
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tabulate
import pprint
import click
import numpy as np
import pandas as pd
from ray.tune.commands import *

import warnings
warnings.filterwarnings("ignore")

def load(experiment_path):
  """ Load a single experiment into a dataframe """  

  experiment_path = os.path.abspath(experiment_path)
  experiment_states = _get_experiment_states(
    experiment_path, exit_on_fail=True)

  # run once per experiment state
  # columns might differ between experiments
  dataframes = []
  for exp_state, exp_name in experiment_states:
    progress, params = _read_experiment(exp_state, experiment_path)
    dataframes.append(_get_value(progress, params, exp_name))

  # concats all dataframes if there are any and return
  if not dataframes: return pd.DataFrame([])
  return pd.concat(dataframes, axis=0, ignore_index=True, sort=False)


def load_many(experiment_paths):
  """ Load several experiments into a single dataframe""" 

  dataframes = [load(path) for path in experiment_paths]
  return pd.concat(dataframes, axis=0, ignore_index=True, sort=False)    

def _read_experiment(experiment_state, experiment_path):
  checkpoint_dicts = experiment_state["checkpoints"]
  checkpoint_dicts = [flatten_dict(g) for g in checkpoint_dicts]

  progress = {}
  params = {}
  # TODO: no real use for exp_directories outside this function, why get it?
  exp_directories = {} 
  for exp in checkpoint_dicts:
    if exp.get("logdir", None) is None:
      continue
    exp_dir = os.path.basename(exp["logdir"])
    exp_tag = exp["experiment_tag"]
    csv = os.path.join(experiment_path, exp_dir, "progress.csv")
    # check if file size is > 0 before proceeding
    if os.stat(csv).st_size:
      progress[exp_tag] = pd.read_csv(csv)
      exp_directories[exp_tag] = os.path.abspath(
        os.path.join(experiment_path, exp_dir))

      # comment out on checkpoint recover for now
      # # Figure out checkpoint file (.pt or .pth) if it exists. For some reason
      # # we need to switch to the directory in order for glob to work.
      # checkpoint_directories = {}
      # ed = os.path.abspath(os.path.join(experiment_path, exp_dir))
      # os.chdir(ed)
      # cds = glob.glob("checkpoint*")
      # if len(cds) > 0:
      #   cd = max(cds)
      #   cf = glob.glob(os.path.join(cd, "*.pt"))
      #   cf += glob.glob(os.path.join(cd, "*.pth"))
      #   if len(cf) > 0:
      #     checkpoint_directories[exp_tag] = os.path.join(
      #       ed, cf[0])
      #   else:
      #     checkpoint_directories[exp_tag] = ""
      # else:
      #   checkpoint_directories[exp_tag] = ""

      # Read in the configs for this experiment
      paramsFile = os.path.join(experiment_path, exp_dir, "params.json")
      with open(paramsFile) as f:
        params[exp_tag] = json.load(f)

  return progress, params

def _get_value(progress, params, exp_name, exp_substring="",
              tags=["test_accuracy", "noise_accuracy", "mean_accuracy"],
              which='max'):
  """
  For every experiment whose name matches exp_substring, scan the history
  and return the appropriate value associated with tag.
  'which' can be one of the following:
      last: returns the last value
       min: returns the minimum value
       max: returns the maximum value
    median: returns the median value
  
  Returns a pandas dataframe with two columns containing name and tag value

  Modified to run once per experiment state
  """

  # Collect experiment names that match exp at all
  exps = [e for e in progress if exp_substring in e]

  # empty histories always return None
  columns = ['Experiment Name']
  
  # add the columns names for main tags
  for tag in tags:
    columns.append(tag)
    columns.append(tag+'_'+which)
    if which in ["max", "min"]:
      columns.append("epoch_"+str(tag))
  columns.append('epochs')
  columns.append('start_learning_rate')
  columns.append('end_learning_rate')
  columns.append('early_stop')
  columns.append('experiment_file_name')
  columns.extend(['trial_time', 'mean_epoch_time'])
  columns.extend(['trial_train_time', 'mean_epoch_train_time'])

  # add the remaining variables
  columns.extend(params[exps[0]].keys())

  all_values = []
  for e in exps:
    # values for the experiment name
    values = [e]
    # values for the main tags
    for tag in tags:
      values.append(progress[e][tag].iloc[-1])
      if which == "max":
        values.append(progress[e][tag].max())
        v = progress[e][tag].idxmax()
        values.append(v)
      elif which == "min":
        values.append(progress[e][tag].min())
        values.append(progress[e][tag].idxmin())
      elif which == "median":
        values.append(progress[e][tag].median())
      elif which == "last":
        values.append(progress[e][tag].iloc[-1])
      else:
        raise RuntimeError("Invalid value for which='{}'".format(which))

    # add remaining main tags
    values.append(progress[e]['training_iteration'].iloc[-1])
    values.append(progress[e]['learning_rate'].iloc[0])
    values.append(progress[e]['learning_rate'].iloc[-1])
    # consider early stop if there is a signal and haven't reached last iteration
    if (params[e]['iterations'] != progress[e]['training_iteration'].iloc[-1] 
        and progress[e]['stop'].iloc[-1]):
      values.append(1)
    else:
      values.append(0)
    values.append(exp_name)
    # store time in minutes
    values.append(progress[e]['epoch_time'].sum()/60)
    values.append(progress[e]['epoch_time'].mean()/60)
    values.append(progress[e]['epoch_time_train'].sum()/60)
    values.append(progress[e]['epoch_time_train'].mean()/60)

    # remaining values
    for v in params[e].values():
      if isinstance(v,list):
        values.append(np.mean(v))
      else:
        values.append(v)         
    
    all_values.append(values)

  p = pd.DataFrame(all_values, columns=columns)
    
  return p


def get_checkpoint_file(exp_substring=""):
  """
  For every experiment whose name matches exp_substring, return the
  full path to the checkpoint file. Returns a list of paths.
  """
  # Collect experiment names that match exp at all
  exps = [e for e in progress if exp_substring in e]

  paths = [self.checkpoint_directories[e] for e in exps]

  return paths

def _get_experiment_states(experiment_path, exit_on_fail=False):
  """
  Return every experiment state JSON file in the path as a list of dicts.
  The list is sorted such that newer experiments appear later.
  """
  experiment_path = os.path.expanduser(experiment_path)
  experiment_state_paths = glob.glob(
    os.path.join(experiment_path, "experiment_state*.json"))

  if not experiment_state_paths:
    print("No experiment state found for experiment {}".format(experiment_path))
    return []

  experiment_state_paths = list(experiment_state_paths)
  experiment_state_paths.sort()
  experiment_states = []
  for experiment_filename in list(experiment_state_paths):
    with open(experiment_filename) as f:
      experiment_states.append((json.load(f), experiment_filename))

  return experiment_states

def get_parameters(sorted_experiments):
  for i,e in sorted_experiments.iterrows():
    if e['Experiment Name'] in params:
      params = params[e['Experiment Name']]
      print(params['cnn_percent_on'][0])

  print('test_accuracy')
  for i,e in sorted_experiments.iterrows():
    print(e['test_accuracy'])

  print('noise_accuracy')
  for i,e in sorted_experiments.iterrows():
    print(e['noise_accuracy'])  

# def best_experiments(min_test_accuracy=0.86, min_noise_accuracy=0.785, sort_by="noise_accuracy"):
#   """
#   Return a dataframe containing all experiments whose best test_accuracy and
#   noise_accuracy are above the specified thresholds.
#   """
#   best_accuracies = self._get_value()
#   best_accuracies.sort_values(sort_by, axis=0, ascending=False,
#                inplace=True, na_position='last')
#   columns = best_accuracies.columns
#   best_experiments = pd.DataFrame(columns=columns)
#   for i, row in best_accuracies.iterrows():
#     if ((row["test_accuracy"] > min_test_accuracy)
#          and (row["noise_accuracy"] > min_noise_accuracy)):
#       best_experiments = best_experiments.append(row)

#   return best_experiments


# def prune_checkpoints(max_test_accuracy=0.86, max_noise_accuracy=0.785):
#   """
#   TODO: delete the checkpoints for all models whose best test_accuracy and
#   noise_accuracy are below the specified thresholds.
#   """
#   pass



