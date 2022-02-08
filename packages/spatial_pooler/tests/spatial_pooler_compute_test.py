# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np
import time 
import unittest

from nupic.research.frameworks.spatial_pooler import SpatialPooler

real_type = np.float32
uint_type = np.uint32


class SpatialPoolerComputeTest(unittest.TestCase):
    '''
    end-to-end test of the compute function
    '''

    def basic_compute_loop(self, sp, input_size, minicolumn_dims):
        '''
        feed in some vectors and retrieve outputs. ensure the right number of minicolumns win and that we always get binary outputs. 
        '''

        num_records = 100
        generator = np.random.default_rng()

        input_matrix = (generator.random((num_records, input_size)) > 0.8).astype(uint_type)

        y = np.zeros(minicolumn_dims, dtype=uint_type)

        # with learning
        for v in input_matrix:
            y.fill(0)
            sp.compute(v, True, y)
            self.assertEqual(sp.num_active_minicolumns_per_inh_area, y.sum())
            self.assertEqual(0, y.min())
            self.assertEqual(1, y.max())

        # without learning
        for v in input_matrix:
            y.fill(0)
            sp.compute(v, False, y)
            self.assertEqual(sp.num_active_minicolumns_per_inh_area, y.sum())
            self.assertEqual(0, y.min())
            self.assertEqual(1, y.max())


    def test_basic_compute1(self):
        '''
        run basic_compute_loop with mostly default parameters.
        '''
        
        input_size = 30
        minicolumn_dims = 50

        sp = SpatialPooler(
            input_dims=[input_size],
            minicolumn_dims=[minicolumn_dims],
            num_active_minicolumns_per_inh_area=10,
            local_density=-1,
            potential_radius=input_size,
            potential_percent=0.5,
            global_inhibition=True,
            stimulus_threshold=0.0,
            synapse_perm_inc=0.05,
            synapse_perm_dec=0.008,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=1000,
            boost_strength=0.0,
            seed=int((time.time() % 10000)*10)
        )

        print('test_basic_compute1, SP seed set to:', sp.seed)

        self.basic_compute_loop(sp, input_size, minicolumn_dims)
    

    def test_basic_compute2(self):
        '''
        run basic_compute_loop with learning turned off. 
        '''
        
        input_size = 100
        minicolumn_dims = 100

        sp = SpatialPooler(
            input_dims=[input_size],
            minicolumn_dims=[minicolumn_dims],
            num_active_minicolumns_per_inh_area=10,
            local_density=-1,
            potential_radius=input_size,
            potential_percent=0.5,
            global_inhibition=True,
            stimulus_threshold=0.0,
            synapse_perm_inc=0.0,
            synapse_perm_dec=0.0,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=1000,
            boost_strength=0.0,
            seed=int((time.time() % 10000)*10)
        )

        print('test_basic_compute2, SP seed set to:', sp.seed)

        self.basic_compute_loop(sp, input_size, minicolumn_dims)


if __name__ == '__main__':
    unittest.main()