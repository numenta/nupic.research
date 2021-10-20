
import argparse
import json
import pandas as pd
import os
import re
from tqdm import tqdm

def load(experiment_path):
    """Load a single experiment into a dataframe"""
    experiment_path = os.path.expanduser(experiment_path)
    experiment_states = _get_experiment_states(experiment_path, exit_on_fail=True)

    # run once per experiment state
    # columns might differ between experiments
    summaries = []
    histories = {}
    all_hyperparams = set()
    for exp_state, exp_name in experiment_states:
        progress, params = _read_experiment(exp_state, experiment_path)
        local_hyperparams = params_to_set(params)
        if len(progress) != 0:
            summaries.append(
                _get_value(progress, params, exp_name)
            )
            all_hyperparams = all_hyperparams.union(local_hyperparams)
            histories.update(progress)

    # concats all dataframes if there are any and return
    if not summaries:
        return pd.DataFrame([]), pd.DataFrame([]), all_hyperparams

    summary_df = pd.concat(summaries, axis=0, ignore_index=True, sort=False)
    return summary_df, histories, all_hyperparams


def param_val_to_float(val):
    """
    Have to process ZeroSegment and SparseMLP cases differently to get weight_sparsity
    and percent_on parameters.
    """

    # Handle the case of tuples
    if (val[0] == "(") or (val[0] == "["):
        float_val = float(val[1:].split(",")[0])
    else:
        float_val = float(val)

    return float_val


def parse_experiment_tag(tag):

    # 8_params... -> params...
    reduced_tag = re.split(r"[0-9]_", tag)[1]
    tag_split = reduced_tag.split(",")
    param_dict = {}
    for param in tag_split:
        if "=" in param:
            param_split = param.split("=")
            parameter, value = param_split
            param_dict[parameter] = param_val_to_float(value)
    return param_dict


def read_result(result_file):
    """
    Designed to load data from a result.json file from a ray tune experiment. Note that
    the mean_accuracy measure is on the last line, so for now just skip to that result.
    """
    with open(result_file, "r") as f:
        for line in f:
            pass
        results = pd.Series(json.loads(line))

    if "mean_accuracy" not in results:
        msg = f"Skipping file {result_file} that did not have mean_accuracy measure "
        print(msg)
        return
    param_dict = parse_experiment_tag(results["experiment_tag"])

    # Unpack any params not logged
    for param, val in param_dict.items():
        results[param] = val

    results["file"] = result_file
    if "weight_sparsity" not in param_dict:
        results["weight_sparsity"] = 0.
    if "kw_percent_on" not in param_dict:
        results["kw_percent_on"] = 1.

    return results


def get_all_results(base_dir):
    results = []
    for dirpath, dirname, files in tqdm(os.walk(base_dir)):
        for file in files:
            if file == "result.json":
                result_file = os.path.join(dirpath, file)
                if os.stat(result_file).st_size == 0:
                    print(f"Skipping empty file: {result_file}")
                    continue
                result = read_result(result_file)
                if result is not None:
                    results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Pass in the path to the outer-most directory for a
            hyperparameter search. This script will agregate results and save
            a csv file.""")

    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="Path to a hyperparameter search run")

    args = parser.parse_args()
    experiment_path = args.directory
    results = get_all_results(experiment_path)
    df = pd.DataFrame(results)

    print(f"\n\n -- Writing results for {len(results)} files to {experiment_path}"
          ".aggregate_results.csv --\n\n")
    df.to_csv(os.path.join(experiment_path, "aggregate_results.csv"))