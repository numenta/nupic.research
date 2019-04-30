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




class ExperimentBrowser(object):

  def __init__(self, experiment_path):
    self.experiment_path = os.path.abspath(experiment_path)
    self.experiment_states = self._get_experiment_states(
      self.experiment_path, exit_on_fail=True)

    self.progress = {}
    self.exp_directories = {}
    self.checkpoint_directories = {}
    self.params = {}
    for experiment_state in self.experiment_states:
      self._read_experiment(experiment_state)


  def _read_experiment(self, experiment_state):
    checkpoint_dicts = experiment_state["checkpoints"]
    checkpoint_dicts = [flatten_dict(g) for g in checkpoint_dicts]

    for exp in checkpoint_dicts:
      if exp.get("logdir", None) is None:
        continue
      exp_dir = os.path.basename(exp["logdir"])
      csv = os.path.join(self.experiment_path, exp_dir, "progress.csv")
      self.progress[exp["experiment_tag"]] = pd.read_csv(csv)
      self.exp_directories[exp["experiment_tag"]] = os.path.abspath(
        os.path.join(self.experiment_path, exp_dir))

      # Figure out checkpoint file (.pt or .pth) if it exists. For some reason
      # we need to switch to the directory in order for glob to work.
      ed = os.path.abspath(os.path.join(self.experiment_path, exp_dir))
      os.chdir(ed)
      cds = glob.glob("checkpoint*")
      if len(cds) > 0:
        cd = max(cds)
        cf = glob.glob(os.path.join(cd, "*.pt"))
        cf += glob.glob(os.path.join(cd, "*.pth"))
        if len(cf) > 0:
          self.checkpoint_directories[exp["experiment_tag"]] = os.path.join(
            ed, cf[0])
        else:
          self.checkpoint_directories[exp["experiment_tag"]] = ""
      else:
        self.checkpoint_directories[exp["experiment_tag"]] = ""

        # Read in the configs for this experiment
        paramsFile = os.path.join(self.experiment_path, exp_dir, "params.json")
        with open(paramsFile) as f:
          self.params[exp["experiment_tag"]] = json.load(f)


  def get_value(self, exp_substring="",
                tags=["test_accuracy", "noise_accuracy"],
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
    """
    # Collect experiment names that match exp at all
    exps = [e for e in self.progress if exp_substring in e]

    # empty histories always return None
    columns = ['name']
    for tag in tags:
      columns.append(tag)
      if which in ["max", "min"]:
        columns.append("epoch_"+str(tag))
    p = pd.DataFrame(columns=columns)
    for e in exps:
      values = [e]
      for tag in tags:
        if which == "max":
          values.append(self.progress[e][tag].max())
          v = self.progress[e][tag].idxmax()
          values.append(v)
        elif which == "min":
          values.append(self.progress[e][tag].min())
          values.append(self.progress[e][tag].idxmin())
        elif which == "median":
          values.append(self.progress[e][tag].median())
        elif which == "last":
          values.append(self.progress[e][tag].iloc[-1])
        else:
          raise RuntimeError("Invalid value for which='{}'".format(which))

      p1 = pd.DataFrame([values], columns=columns)
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


  def _get_experiment_states(self, experiment_path, exit_on_fail=False):
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


@click.command()
@click.argument("experiment_path", required=True, type=str)
@click.option('--name', default="", help='The substring to match')
@click.option('--tag', default="test_accuracy", help='The tag to extract and sort by')
@click.option('--which', default="max", help='The function to use for extracting')
def summarize_trials(experiment_path, name, tag, which):
    """Summarizes trials in the directory subtree starting at the given path."""
    browser = ExperimentBrowser(experiment_path)
    tags = ["test_accuracy", "noise_accuracy"]
    if tag not in tags: tags.append(tag)
    p = browser.get_value(exp_substring=name, tags=tags, which=which)
    p.sort_values(tag, axis=0, ascending=False,
                 inplace=True, na_position='last')
    print_df_tabular(p)
    # print("Checkpoints:")
    # pprint.pprint(browser.get_checkpoint_file(name))


if __name__ == "__main__":
    summarize_trials()
