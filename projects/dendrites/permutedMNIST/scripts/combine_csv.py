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

import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Pass in the path to the outer-most directory for a
            hyperparameter search. This script will agregate results and save
            a csv file.""")

    parser.add_argument("-A", "--aggregate", type=str, nargs="+",
                        required=True,
                        help="List of files to aggregate")
    parser.add_argument("-O", "--output", type=str,
                        help="Where to send the output to")

    args = parser.parse_args()
    if args.aggregate:
        df_list = []
        for file in args.aggregate:
            data = pd.read_csv(file)
            df_list.append(data)

        big_df = pd.concat(df_list)
        big_df.to_csv(args.output)
