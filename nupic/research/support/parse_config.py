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

import configparser

def parse_config(config_file, experiments=None, globals=None, locals=None):
  """
  Parse configuration file optionally filtering for specific experiments/sections
  :param config_file: Configuration file
  :param experiments: Optional list of experiments
  :param globals: global symbol table to use during `eval`
  :param locals: local symbol table to use during `eval`
  :return: Dictionary with the parsed configuration
  """
  cfgparser = configparser.ConfigParser()
  cfgparser.read_file(config_file)

  params = {}
  for exp in cfgparser.sections():
    if not experiments or exp in experiments:
      values = dict(cfgparser.defaults())
      values.update(dict(cfgparser.items(exp)))
      item = {}
      for k, v in values.items():
        try:
          item[k] = eval(v, globals, locals)
        except (NameError, SyntaxError):
          item[k] = v

      params[exp] = item

  return params


