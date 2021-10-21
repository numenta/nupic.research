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
