#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import argparse
from pprint import pprint

from nupic.research.frameworks.sigopt import SigOptExperiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple command line utility for SigOpt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument("id", type=int,
                        help="The experiment id")
    parser.add_argument("-p", "--properties", action="store_true",
                        help="Print out the properties for this experiment")
    parser.add_argument("-o", "--observations", action="store_true",
                        help="List observations from this experiment")
    parser.add_argument("-os", "--open-suggestions", action="store_true",
                        help="List open suggestions from this experiment")
    parser.add_argument("-do", "--delete-open-suggestions", action="store_true",
                        help="Delete open suggestions from this experiment")

    args = parser.parse_args()

    s = SigOptExperiment(args.id)

    if "observations" in args:
        observations = s.observations()
        pprint(observations)

    if "open_suggestions" in args:
        su = s.open_suggestions()
        print("There are", len(su), "open suggestions")
        pprint(su)

    if "delete_open_suggestions" in args:
        s.delete_open_suggestions()

    if "properties" in args:
        pprint(s.get_experiment_details(), width=100)
