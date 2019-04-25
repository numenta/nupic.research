from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tabulate
import pprint
import click
from ray.tune.commands import _get_experiment_state
from ray.tune.commands import *

def print_df_tabular(df):
  headers = list(df.columns)
  headers.remove("name")
  headers.insert(0, "name")

  table = [headers]
  for i, row in df.iterrows():
    table.append([row[n] for n in headers])
  print(tabulate(table, headers="firstrow", tablefmt="grid"))


def get_experiment_states(experiment_path, exit_on_fail=False):
  """
  Return every experiment state JSON file in the path as a list of dicts.
  The list is sorted such that newer experiments appear later.
  """
  experiment_path = os.path.expanduser(experiment_path)
  experiment_state_paths = glob.glob(
    os.path.join(experiment_path, "experiment_state*.json"))
  if not experiment_state_paths:
    if exit_on_fail:
      print("No experiment state found!")
      sys.exit(0)
    else:
      return

  experiment_state_paths = list(experiment_state_paths)
  experiment_state_paths.sort()
  experiment_states = []
  for experiment_filename in list(experiment_state_paths):

    with open(experiment_filename) as f:
      experiment_states.append(json.load(f))

  return experiment_states


class ExperimentBrowser(object):

  def __init__(self, experiment_path):
    self.experiment_path = os.path.abspath(experiment_path)
    self.experiment_states = get_experiment_states(
      self.experiment_path, exit_on_fail=True)

    self.progress = {}
    self.exp_directories = {}
    self.checkpoint_directories = {}
    for experiment_state in self.experiment_states:
      self._read_experiment(experiment_state)


  def _read_experiment(self, experiment_state):
    print("Examining experiments in ",
          experiment_state["runner_data"]["_session_str"])
    checkpoint_dicts = experiment_state["checkpoints"]
    checkpoint_dicts = [flatten_dict(g) for g in checkpoint_dicts]

    for exp in checkpoint_dicts:
      exp_dir = os.path.basename(exp["logdir"])
      csv = os.path.join(self.experiment_path, exp_dir, "progress.csv")
      self.progress[exp["experiment_tag"]] = pd.read_csv(csv)
      self.exp_directories[exp["experiment_tag"]] = os.path.abspath(
        os.path.join(self.experiment_path, exp_dir))

      # Figure out checkpoint file if it exists. For some reason we need to
      # switch to the directory in order for glob to work.
      ed = os.path.abspath(os.path.join(self.experiment_path, exp_dir))
      os.chdir(ed)
      cd = max(glob.glob("checkpoint*"))
      cf = glob.glob(os.path.join(cd, "*.pt"))
      if len(cf) > 0:
        self.checkpoint_directories[exp["experiment_tag"]] = os.path.abspath(cf[0])
      else:
        self.checkpoint_directories[exp["experiment_tag"]] = ""


  def get_value(self, exp_substring="", tag="mean_accuracy", which='max'):
    """
    For every experiment whose name matches exp_substring, scan the history
    and return the appropriate value associated with tag.
    'which' can be one of the following:
        last: returns the last value
         min: returns the minimum value
         max: returns the maximum value
      median: returns the median value

    Returns a pandas dataframe with two columns containing name and tag value
    """
    # Collect experiment names that match exp at all
    exps = [e for e in self.progress if exp_substring in e]

    # empty histories always return None
    p = pd.DataFrame(columns=['name', tag])
    for e in exps:
      if which == "max":
        v = self.progress[e][tag].max()
      elif which == "min":
        v = self.progress[e][tag].min()
      elif which == "median":
        v = self.progress[e][tag].median()
      elif which == "last":
        v = self.progress[e][tag].iloc[-1]
      else:
        raise RuntimeError("Invalid value for which='{}'".format(which))

      p1 = pd.DataFrame([[e, v]], columns=['name', tag])
      p = p.append(p1)

    return p

  def get_checkpoint_file(self, exp_substring=""):
    """
    For every experiment whose name matches exp_substring, return the
    full path to the checkpoint file. Returns a list of paths.
    """
    # Collect experiment names that match exp at all
    exps = [e for e in self.progress if exp_substring in e]

    paths = [self.checkpoint_directories[e] for e in exps]

    return paths


@click.command()
@click.argument("experiment_path", required=True, type=str)
@click.option('--name', default="", help='The substring to match')
@click.option('--tag', default="mean_accuracy", help='The tag to extract')
@click.option('--which', default="max", help='The function to use for extracting')
def summarize_trials(experiment_path, name, tag, which):
    """Summarizes trials in the directory subtree starting at the given path."""
    browser = ExperimentBrowser(experiment_path)
    p = browser.get_value(exp_substring=name, tag=tag, which=which)
    p.sort_values(tag, axis=0, ascending=False,
                 inplace=True, na_position='last')
    print_df_tabular(p)
    print("Checkpoints:")
    pprint.pprint(browser.get_checkpoint_file(name))


if __name__ == "__main__":
    summarize_trials()
