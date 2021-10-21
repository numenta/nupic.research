import ast
import collections
import matplotlib.pyplot as plt
import numpy as np
import os

"""
Code intended to read and analyze csvs that are the result of either
aggregate_results.sh or aggregate_results.py
"""

def parse_configs(X):
    """
    Use ast to parse config column. Json will not work due to the mixture
    of single and double quotes. Returns dataframe with only config info.
    Helper function for read_and_parse_csv.
    """
    flat_dicts = [flatten(ast.literal_eval(Xiloc[i]["config"]))
                    for i in range(Xshape[0])]
    config_df = pd.DataFrame(flat_dicts)
    return config_df

def flatten(d, parent_key='', sep='_'):
    """
    Helper for parsing config column. AST parser returns nested dictionary.
    This flattens dictionary by prepending parent keys to full key name.
    Note that columns like "num_tasks" get set to dataset_args_num_tasks due
    to the nestedness of the config.

    Source: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys  # noqa E501
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_and_parse_csv(csv_path):
    """
    Called read and parse because it reads the csv, also parses the config
    column into a data frame, and merges the dataframes.
    """

    data = pd.read_csv(csv_path)
    config_dict = parse_configs(data)

    # Dataframes are the same size. Re-index so you can merge.
    data.reset_index(inplace=True)
    config_dict.reset_index(inplace=True)

    # Merge dataframes
    X = pd.merge(data, config_dict)

    return X
