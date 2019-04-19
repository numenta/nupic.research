from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
from ray.tune.commands import _get_experiment_state
from ray.tune.commands import *



class experimentBrowser(object):

  def __init__(self, experiment_path):
    # list of keys, that had to be renamed because they contained spaces
    self.experiment_path = os.path.abspath(experiment_path)
    self.experiment_state = _get_experiment_state(
      self.experiment_path, exit_on_fail=True)
    checkpoint_dicts = self.experiment_state["checkpoints"]
    checkpoint_dicts = [flatten_dict(g) for g in checkpoint_dicts]
    self.progress = {}
    for exp in checkpoint_dicts:
      csv = os.path.join(exp["logdir"], "progress.csv")
      self.progress[exp["experiment_tag"]] = pd.read_csv(csv)
      print(exp["experiment_tag"],
            self.progress[exp["experiment_tag"]].mean_accuracy.max()
            )

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


@click.command()
@click.argument("experiment_path", required=True, type=str)
@click.option('--name', default="", help='The substring to match')
@click.option('--tag', default="mean_accuracy", help='The tag to extract')
@click.option('--which', default="max", help='The function to use for extracting')
def summarize_trials(experiment_path, name, tag, which):
    """Summarizes trials in the directory subtree starting at the given path."""
    browser = experimentBrowser(experiment_path)
    p = browser.get_value(exp_substring=name, tag=tag, which=which)
    p.sort_values(tag, axis=0, ascending=False,
                 inplace=True, na_position='last')
    print(p)


if __name__ == "__main__":
    summarize_trials()
