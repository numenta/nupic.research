#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

# Script used to convert stdout test results to CSV
# The last test batch is in the following format:
# "Test: [390/391] Time  1.146 ( 1.062)    Loss 1.1572e+01 (9.2752e+00)    Acc@1   0.00 (  6.77)   Acc@5   1.25 ( 18.63)"
# The values in parenthesis represent the mean over all batches

echo "epoch,loss,top1,top5"
cat -| grep '390/391' | awk -F"[()]" '{print NR","$4","$6","$8}'
