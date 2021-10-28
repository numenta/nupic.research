#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import ast
import collections

import pandas as pd


"""
Code intended to read and analyze csvs that are the result of either
aggregate_results.sh or aggregate_results.py
"""


def parse_configs(x):
    """
    Use ast to parse config column. Json will not work due to the mixture
    of single and double quotes. Returns dataframe with only config info.
    Helper function for read_and_parse_csv.
    """
    flat_dicts = [flatten(ast.literal_eval(x.iloc[i]["config"]))
                  for i in range(x.shape[0])]
    config_df = pd.DataFrame(flat_dicts)
    return config_df


def flatten(d, parent_key="", sep="_"):
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
    X = pd.merge(data, config_dict)  # noqa N806 (capital X is a matrix)

    return X
