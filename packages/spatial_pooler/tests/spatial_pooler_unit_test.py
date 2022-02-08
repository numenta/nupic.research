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
from unittest.mock import Mock
from copy import copy

from nupic.research.frameworks.spatial_pooler import SpatialPooler

real_type = np.float32
uint_type = np.uint32

SEED = int((time.time() % 10000) * 10)


class SpatialPoolerUnitTest(unittest.TestCase):
    ''' 
    unit tests for SpatialPooler class. 
    '''

    def setUp(self):
        '''
        create spatial pooler's base parameters.
        '''

        self.base_params = {
            'input_dims' : [5],
            'minicolumn_dims' : [5],
            'num_active_minicolumns_per_inh_area' : 3,
            'local_density' : -1,
            'potential_radius' : 5,
            'potential_percent' : 0.5,
            'global_inhibition' : False,
            'stimulus_threshold' : 0.0,
            'synapse_perm_inc' : 0.1,
            'synapse_perm_dec' : 0.01,
            'synapse_perm_connected' : 0.1,
            'min_percent_overlap_duty_cycles' : 0.1,
            'duty_cycle_period' : 10,
            'boost_strength' : 10.0,
            'seed' : SEED
        }


    def test_compute1(self):
        '''
        check that feeding in same input vectors leads to polarized permanence values: either zeros or ones but no fractions
        '''

        sp = SpatialPooler(
            input_dims=[9],
            minicolumn_dims=[5],
            num_active_minicolumns_per_inh_area=3,
            local_density=-1,
            potential_radius=3,
            potential_percent=0.5,
            global_inhibition=False,
            stimulus_threshold=1,
            synapse_perm_inc=0.1,
            synapse_perm_dec=0.1,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.1,
            duty_cycle_period=10,
            boost_strength=10.0,
            seed=SEED
        )

        sp.potential_pools = np.ones((sp.num_minicolumns, sp.num_inputs))

        # mock that all minicolumns have won during the local inhibition process
        sp.inhibit_minicolumns = Mock(return_value=np.array(range(5)))

        input_vector = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1])
        active_array = np.zeros(5)

        for i in range(20):
            sp.compute(input_vector, True, active_array)
        
        for i in range(sp.num_minicolumns):
            permanence = sp.get_permanences()[i, :]

            self.assertEqual(list(permanence), list(input_vector))
        

    def test_compute2(self):
        '''
        check that minicolumns only change the permanence values for inputs that are within their potential pool
        '''

        sp = SpatialPooler(
            input_dims=[10],
            minicolumn_dims=[5],
            num_active_minicolumns_per_inh_area=3,
            local_density=-1,
            potential_radius=3,
            potential_percent=0.5,
            global_inhibition=False,
            stimulus_threshold=1,
            synapse_perm_inc=0.1,
            synapse_perm_dec=0.01,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.1,
            duty_cycle_period=10,
            boost_strength=10.0,
            seed=SEED
        )

        # mock that all minicolumns have won during the local inhibition process
        self.inhibit_minicolumns = Mock(return_value=np.array((range(5))))

        input_vector = np.ones(sp.num_inputs)
        active_array = np.zeros(5)

        for i in range(20):
            sp.compute(input_vector, True, active_array)

        for i in range(sp.num_minicolumns):
            potential = sp.get_potential_pools()[i, :]
            permanence = sp.get_permanences()[i, :]

            self.assertEqual(list(potential), list(permanence))


    def test_ZeroOverlap_NoStimulusThreshold_GlobalInhibition(self):
        ''' 
        when stimulus_threshold is 0, allow minicolumns without any overlap to become active. focuses on global inhibition.
        '''

        input_size = 10
        num_minicolumns = 20

        sp = SpatialPooler(
            input_dims=[input_size],
            minicolumn_dims=[num_minicolumns],
            num_active_minicolumns_per_inh_area=3,
            potential_radius=10,
            global_inhibition=True,
            stimulus_threshold=0,
            seed=SEED
        )

        input_vector = np.zeros(input_size)
        active_array = np.zeros(num_minicolumns)

        sp.compute(input_vector, True, active_array)

        self.assertEqual(len(active_array.nonzero()[0]), 3)


    def test_ZeroOverlap_StimulusThreshold_GlobalInhibition(self):
        ''' 
        when stimulus_threshold is > 0, don't allow minicolumns without any overlap to become active. focuses on global inhibition.
        '''

        input_size = 10
        num_minicolumns = 20

        sp = SpatialPooler(
            input_dims=[input_size],
            minicolumn_dims=[num_minicolumns],
            num_active_minicolumns_per_inh_area=3,
            potential_radius=10,
            global_inhibition=True,
            stimulus_threshold=1,
            seed=SEED
        )

        input_vector = np.zeros(input_size)
        active_array = np.zeros(num_minicolumns)

        sp.compute(input_vector, True, active_array)

        self.assertEqual(len(active_array.nonzero()[0]), 0)

    
    def test_ZeroOverlap_NoStimulusThreshold_LocalInhibition(self):
        '''
        when stimulus_threshold is 0, allow minicolumns without any overlap to become active. focuses on local inhibition.
        '''
        
        input_size = 10
        num_minicolumns = 20

        sp = SpatialPooler(
            input_dims=[input_size],
            minicolumn_dims=[num_minicolumns],
            num_active_minicolumns_per_inh_area=1,
            potential_radius=5,
            global_inhibition=False,
            stimulus_threshold=0,
            seed=SEED
        )

        sp.set_inhibition_radius(2)

        input_vector = np.zeros(input_size)
        active_array = np.zeros(num_minicolumns)

        sp.compute(input_vector, True, active_array)

        self.assertEqual(len(active_array.nonzero()[0]), 7)

    
    def test_ZeroOverlap_StimulusThreshold_LocalInhibition(self):
        ''' 
        when stimulus threshold is > 0, don't allow minicolumns without any overlap to become active. focuses on local inhibition.
        '''

        input_size = 10
        num_minicolumns = 20

        sp = SpatialPooler(
            input_dims=[input_size],
            minicolumn_dims=[num_minicolumns],
            num_active_minicolumns_per_inh_area=3,
            potential_radius=10,
            global_inhibition=False,
            stimulus_threshold=1,
            seed=SEED
        )

        input_vector = np.zeros(input_size)
        active_array = np.zeros(num_minicolumns)

        sp.compute(input_vector, True, active_array)

        self.assertEqual(len(active_array.nonzero()[0]), 0)

    
    def test_overlaps_output(self):
        ''' 
        check that overlaps and boosted_overlaps are correctly returned.
        '''

        sp = SpatialPooler(
            input_dims=[5],
            minicolumn_dims=[3],
            num_active_minicolumns_per_inh_area=5,
            potential_radius=5,
            synapse_perm_inc=0.1,
            synapse_perm_dec=0.1,
            global_inhibition=True,
            seed=1
        )
        
        input_vector = np.ones(5)
        active_array = np.zeros(3)

        expected_output = np.array([2, 1, 1], dtype=real_type)

        boost_factors = 2.0 * np.ones(3)
        sp.set_boost_factors(boost_factors)

        sp.compute(input_vector, True, active_array)

        overlaps = sp.get_overlaps()
        boosted_overlaps = sp.get_boosted_overlaps()

        for i in range(sp.get_num_minicolumns()):
            self.assertEqual(overlaps[i], expected_output[i])
            self.assertEqual(boosted_overlaps[i], (2 * expected_output[i]))

    
    def test_exact_output(self):
        '''
        given a specific input and initialization, SP should return this exact output.
        '''

        expected_output = [
            101, 115, 149, 180, 252, 328, 391, 433, 577, 612, 700, 799, 805, 861, 864, 
            1122, 1161, 1252, 1336, 1352, 1498, 1576, 1584, 1600, 1688, 1731, 1786,
            1793, 1850, 1872, 1975, 1980, 1983, 1996, 2011, 2016, 2021, 2037, 2043, 
            2046
        ]

        sp = SpatialPooler(
            input_dims=(1, 188),
            minicolumn_dims=(2048, 1),
            num_active_minicolumns_per_inh_area=40,
            local_density=-1.0,
            potential_radius=94,
            potential_percent=0.5,
            global_inhibition=True,
            stimulus_threshold=0,
            synapse_perm_inc=0.1,
            synapse_perm_dec=0.01,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=1000,
            boost_strength=10.0,
            seed=1956
        )

        input_vector = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        input_vector = np.array(input_vector, dtype=real_type)

        active_array = np.zeros(2048)

        sp.compute(input_vector, True, active_array)

        sp_output = [i for i, v in enumerate(active_array) if v != 0]

        self.assertEqual(sorted(sp_output), sorted(expected_output))


    def test_strip_never_learned(self):
        '''
        verify that the expected set of unlearned minicolumns matches the computed result.
        '''

        def strip_unlearned_minicolumns(active_duty_cycles, active_array):
            '''
            remove set of minicolumns that have never been active from set of active minicolumns.
            '''

            never_learned = np.where(active_duty_cycles == 0)[0]
            active_array[never_learned] = 0


        sp = SpatialPooler(**self.base_params)

        sp.active_duty_cycles = np.array([0.5, 0.1, 0, 0.2, 0.4, 0])
        active_array = np.array([1, 1, 1, 0, 1, 0])
        strip_unlearned_minicolumns(sp.get_active_duty_cycles(), active_array)
        stripped = np.where(active_array == 1)[0]
        true_stripped = [0, 1, 4]
        self.assertListEqual(true_stripped, list(stripped))

        sp.active_duty_cycles = np.array([0.9, 0, 0, 0, 0.4, 0.3])
        active_array = np.ones(6)
        strip_unlearned_minicolumns(sp.get_active_duty_cycles(), active_array)
        stripped = np.where(active_array == 1)[0]
        true_stripped = [0, 4, 5]
        self.assertListEqual(true_stripped, list(stripped))

        sp.active_duty_cycles = np.array([0, 0, 0, 0, 0, 0])
        active_array = np.ones(6)
        strip_unlearned_minicolumns(sp.get_active_duty_cycles(), active_array)
        stripped = np.where(active_array == 1)[0]
        true_stripped = []
        self.assertListEqual(true_stripped, list(stripped))

        sp.active_duty_cycles = np.ones(6)
        active_array = np.ones(6)
        strip_unlearned_minicolumns(sp.get_active_duty_cycles(), active_array)
        stripped = np.where(active_array == 1)[0]
        true_stripped = range(6)
        self.assertListEqual(list(true_stripped), list(stripped))


    def test_map_minicolumn(self):
        params = self.base_params.copy()

        # test 1D
        params.update({
            'input_dims' : [12],
            'minicolumn_dims' : [4]
        })
        sp = SpatialPooler(**params)

        self.assertEqual(sp.map_minicolumn(0), 1)
        self.assertEqual(sp.map_minicolumn(1), 4)
        self.assertEqual(sp.map_minicolumn(2), 7)
        self.assertEqual(sp.map_minicolumn(3), 10)


        # test 1D with same dimensions of minicolumns and inputs
        params.update({
            'input_dims' : [4],
            'minicolumn_dims' : [4]
        })
        sp = SpatialPooler(**params)

        self.assertEqual(sp.map_minicolumn(0), 0)
        self.assertEqual(sp.map_minicolumn(1), 1)
        self.assertEqual(sp.map_minicolumn(2), 2)
        self.assertEqual(sp.map_minicolumn(3), 3)


        # test 1D with dimensions of length 1
        params.update({
            'input_dims' : [1],
            'minicolumn_dims' : [1]
        })
        sp = SpatialPooler(**params)

        self.assertEqual(sp.map_minicolumn(0), 0)


        # test 2D
        params.update({
            'input_dims' : [36, 12],
            'minicolumn_dims' : [12, 4]
        })
        sp = SpatialPooler(**params)

        self.assertEqual(sp.map_minicolumn(0), 13)
        self.assertEqual(sp.map_minicolumn(4), 49)
        self.assertEqual(sp.map_minicolumn(5), 52)
        self.assertEqual(sp.map_minicolumn(7), 58)
        self.assertEqual(sp.map_minicolumn(47), 418)


        # test 2D with some input dimensions smaller than minicolumn dimensions
        params.update({
            'input_dims' : [3, 5],
            'minicolumn_dims' : [4, 4]
        })
        sp = SpatialPooler(**params)

        self.assertEqual(sp.map_minicolumn(0), 0)
        self.assertEqual(sp.map_minicolumn(3), 4)
        self.assertEqual(sp.map_minicolumn(15), 14)


    def test_map_potential_1D(self):
        params = self.base_params.copy()

        params.update({
            'input_dims' : [12],
            'minicolumn_dims' : [4],
            'potential_radius' : 2,
        })

        params['potential_percent'] = 1
        sp = SpatialPooler(**params)

        expected_mask = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        mask = sp.map_potential(0)
        self.assertListEqual(mask.tolist(), expected_mask)

        expected_mask = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
        mask = sp.map_potential(2)
        self.assertListEqual(mask.tolist(), expected_mask)


    def test_map_potential_2D(self):
        params = self.base_params.copy()

        params.update({
            'input_dims': [6, 12],
            'minicolumn_dims': [2, 4],
            'potential_radius': 1,
            'potential_percent': 1,
        })

        sp = SpatialPooler(**params)

        true_indices = [
            0, 12, 24,
            1, 13, 25,
            2, 14, 26
        ]
        mask = sp.map_potential(0)
        self.assertSetEqual(set(np.flatnonzero(mask).tolist()), set(true_indices))

        true_indices = [
            6, 18, 30,
            7, 19, 31,
            8, 20, 32
        ]
        mask = sp.map_potential(2)
        self.assertSetEqual(set(np.flatnonzero(mask).tolist()), set(true_indices))


    def test_map_potential_1_minicolumn_1_input(self):
        params = self.base_params.copy()

        params.update({
            'input_dims': [1],
            'minicolumn_dims': [1],
            'potential_radius': 2,
            'potential_percent' : 1,
        })

        sp = SpatialPooler(**params)

        expected_mask = [1]
        mask = sp.map_potential(0)
        self.assertListEqual(mask.tolist(), expected_mask)


    def test_inhibit_minicolumns(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.inhibit_minicolumns_global = Mock(return_value = 1)
        sp.inhibit_minicolumns_local = Mock(return_value = 2)
        sp.num_minicolumns = 5
        sp.inhibition_radius = 10
        sp.minicolumn_dims = [5]
        overlaps = sp.generator.choice(sp.num_minicolumns)

        sp.inhibit_minicolumns_global.reset_mock()
        sp.inhibit_minicolumns_local.reset_mock()
        sp.num_active_minicolumns_per_inh_area = 5
        sp.local_density = 0.1
        sp.global_inhibition = True
        sp.inhibition_radius = 5
        true_density = sp.local_density
        sp.inhibit_minicolumns(overlaps)
        self.assertEqual(True, sp.inhibit_minicolumns_global.called)
        self.assertEqual(False, sp.inhibit_minicolumns_local.called)
        density = sp.inhibit_minicolumns_global.call_args[0][1]
        self.assertEqual(true_density, density)

        sp.inhibit_minicolumns_global.reset_mock()
        sp.inhibit_minicolumns_local.reset_mock()
        sp.num_minicolumns = 500
        sp.minicolumn_dims = np.array([50, 10])
        sp.num_active_minicolumns_per_inh_area = -1
        sp.local_density = 0.1
        sp.global_inhibition = False
        sp.inhibition_radius = 7
        true_density = sp.local_density
        overlaps = sp.generator.choice(sp.num_minicolumns)
        sp.inhibit_minicolumns(overlaps)
        self.assertEqual(False, sp.inhibit_minicolumns_global.called)
        self.assertEqual(True, sp.inhibit_minicolumns_local.called)
        self.assertEqual(true_density, density)

        # test translation of num_minicolumns_per_inh_area into local area density
        sp.num_minicolumns = 1000
        sp.minicolumn_dims = np.array([100, 10])
        sp.inhibit_minicolumns_global.reset_mock()
        sp.inhibit_minicolumns_local.reset_mock()
        sp.num_active_minicolumns_per_inh_area = 3
        sp.local_density = -1
        sp.global_inhibition = False
        sp.inhibition_radius = 4
        true_density = 3.0/81.0
        overlaps = sp.generator.choice(sp.num_minicolumns)
        # 3.0 / (((2*4) + 1) ** 2)
        sp.inhibit_minicolumns(overlaps)
        self.assertEqual(False, sp.inhibit_minicolumns_global.called)
        self.assertEqual(True, sp.inhibit_minicolumns_local.called)
        density = sp.inhibit_minicolumns_local.call_args[0][1]
        self.assertEqual(true_density, density)

        # test clipping of local area density to 0.5
        sp.num_minicolumns = 1000
        sp.minicolumn_dims = np.array([100, 10])
        sp.inhibit_minicolumns_global.reset_mock()
        sp.inhibit_minicolumns_local.reset_mock()
        sp.num_active_minicolumns_per_inh_area = 7
        sp.local_density = -1
        sp.global_inhibition = False
        sp.inhibition_radius = 1
        true_density = 0.5
        overlaps = sp.generator.choice(sp.num_minicolumns)
        sp.inhibit_minicolumns(overlaps)
        self.assertEqual(False, sp.inhibit_minicolumns_global.called)
        self.assertEqual(True, sp.inhibit_minicolumns_local.called)
        density = sp.inhibit_minicolumns_local.call_args[0][1]
        self.assertEqual(true_density, density)

    
    def test_update_inhibition_radius(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        # test global inhibition case
        sp.global_inhibition = True
        sp.minicolumn_dims = np.array([57, 31, 2])
        sp.update_inhibition_radius()
        self.assertEqual(sp.inhibition_radius, 57)

        sp.global_inhibition = False
        sp.average_connected_synapses_per_minicolumn = Mock(return_value = 3)
        sp.average_minicolumns_per_input = Mock(return_value = 4)
        true_inhibition_radius = 6
        # ((3 * 4) - 1) / 2 => round up
        sp.update_inhibition_radius()
        self.assertEqual(true_inhibition_radius, sp.inhibition_radius)

        # test clipping at 1.0
        sp.global_inhibition = False
        sp.average_connected_synapses_per_minicolumn = Mock(return_value = 0.5)
        sp.average_minicolumns_per_input = Mock(return_value = 1.2)
        true_inhibition_radius = 1
        sp.update_inhibition_radius()
        self.assertEqual(true_inhibition_radius, sp.inhibition_radius)

        # test rounding up
        sp.global_inhibition = False
        sp.average_connected_synapses_per_minicolumn = Mock(return_value = 2.4)
        sp.average_minicolumns_per_input = Mock(return_value = 2)
        true_inhibition_radius = 2
        # ((2 * 2.4) - 1) / 2.0 => round up
        sp.update_inhibition_radius()
        self.assertEqual(true_inhibition_radius, sp.inhibition_radius)


    def test_average_minicolumns_per_input(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.minicolumn_dims = np.array([2, 2, 2, 2])
        sp.input_dims = np.array([4, 4, 4, 4])
        self.assertEqual(sp.average_minicolumns_per_input(), 0.5)

        sp.minicolumn_dims = np.array([2, 2, 2, 2])
        sp.input_dims = np.array(     [7, 5, 1, 3])
                                   #  2/7 0.4 2 0.666
        true_average_minicolumns_per_input = (2.0/7 + 2.0/5 + 2.0/1 + 2/3.0) / 4
        self.assertEqual(sp.average_minicolumns_per_input(), true_average_minicolumns_per_input)

        sp.minicolumn_dims = np.array([3, 3])
        sp.input_dims = np.array(     [3, 3])
                                   #   1  1
        true_average_minicolumns_per_input = 1
        self.assertEqual(sp.average_minicolumns_per_input(), true_average_minicolumns_per_input)

        sp.minicolumn_dims = np.array([25])
        sp.input_dims = np.array(     [5])
                                   #   5
        true_average_minicolumns_per_input = 5
        self.assertEqual(sp.average_minicolumns_per_input(), true_average_minicolumns_per_input)

        sp.minicolumn_dims = np.array([3, 3, 3, 5, 5, 6, 6])
        sp.input_dims = np.array(     [3, 3, 3, 5, 5, 6, 6])
                                   #   1  1  1  1  1  1  1
        true_average_minicolumns_per_input = 1
        self.assertEqual(sp.average_minicolumns_per_input(), true_average_minicolumns_per_input)

        sp.minicolumn_dims = np.array([3, 6, 9, 12])
        sp.input_dims = np.array(     [3, 3, 3 , 3])
                                   #   1  2  3   4
        true_average_minicolumns_per_input = 2.5
        self.assertEqual(sp.average_minicolumns_per_input(), true_average_minicolumns_per_input)


    def test_average_connected_span_for_minicolumn_1D(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.num_minicolumns = 9
        sp.minicolumn_dims = np.array([9])
        sp.input_dims = np.array([12])
        sp.connected_synapses = np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1]
            ]
        )

        true_average_connected_span = [7, 5, 1, 5, 0, 2, 3, 3, 8]
        
        for i in range(sp.num_minicolumns):
            connected_span = sp.average_connected_synapses_per_minicolumn(i)
            self.assertEqual(true_average_connected_span[i], connected_span)


    def test_average_connected_span_for_minicolumn_2D(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.num_minicolumns = 9
        sp.minicolumn_dims = np.array([9])
        sp.num_inputs = 8
        sp.input_dims = np.array([8])
        sp.connected_synapses = np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1]
            ]
        )

        true_average_connected_span = [7, 5, 1, 5, 0, 2, 3, 3, 8]

        for i in range(sp.num_minicolumns):
            connected_span = sp.average_connected_synapses_per_minicolumn(i)
            self.assertEqual(true_average_connected_span[i], connected_span)

        sp.num_minicolumns = 7
        sp.minicolumn_dims = np.array([7])
        sp.num_inputs = 20
        sp.input_dims = np.array([5, 4])
        sp.connected_synapses = np.zeros((sp.num_minicolumns, sp.num_inputs))

        connected = np.array([
            [[0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
            # rowspan = 3, colspan = 3, avg = 3

            [[1, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
            # rowspan = 2 colspan = 4, avg = 3

            [[1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]],
            # row span = 5, colspan = 4, avg = 4.5

            [[0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]],
            # rowspan = 5, colspan = 1, avg = 3

            [[0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
            # rowspan = 1, colspan = 4, avg = 2.5

            [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]],
            # rowspan = 2, colspan = 2, avg = 2

            [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
            # rowspan = 0, colspan = 0, avg = 0
        ])

        true_average_connected_span = [3, 3, 4.5, 3, 2.5, 2, 0]
        for column_index in range(sp.num_minicolumns):
            sp.connected_synapses[column_index, :] = connected[column_index,:].reshape(-1)

        for i in range(sp.num_minicolumns):
            connectedSpan = sp.average_connected_synapses_per_minicolumn(i)
            self.assertEqual(true_average_connected_span[i], connectedSpan)


    def test_average_connected_span_for_minicolumn_ND(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.input_dims = np.array([4, 4, 2, 5])
        sp.num_inputs = np.prod(sp.input_dims)
        sp.num_minicolumns = 5
        sp.minicolumn_dims = np.array([5])
        sp.connected_synapses = np.zeros((sp.num_minicolumns, sp.num_inputs))

        connected = np.zeros(sp.num_inputs).reshape(sp.input_dims)
        connected[1][0][1][0] = 1
        connected[1][0][1][1] = 1
        connected[3][2][1][0] = 1
        connected[3][0][1][0] = 1
        connected[1][0][1][3] = 1
        connected[2][2][1][0] = 1
        # span:   3  3  1  4, avg = 11/4
        sp.connected_synapses[0,:] = connected.reshape(-1)

        connected = np.zeros(sp.num_inputs).reshape(sp.input_dims)
        connected[2][0][1][0] = 1
        connected[2][0][0][0] = 1
        connected[3][0][0][0] = 1
        connected[3][0][1][0] = 1
        # spn:    2  1  2  1, avg = 6/4
        sp.connected_synapses[1,:] = connected.reshape(-1)

        connected = np.zeros(sp.num_inputs).reshape(sp.input_dims)
        connected[0][0][1][4] = 1
        connected[0][0][0][3] = 1
        connected[0][0][0][1] = 1
        connected[1][0][0][2] = 1
        connected[0][0][1][1] = 1
        connected[3][3][1][1] = 1
        # span:   4  4  2  4, avg = 14/4
        sp.connected_synapses[2,:] = connected.reshape(-1)

        connected = np.zeros(sp.num_inputs).reshape(sp.input_dims)
        connected[3][3][1][4] = 1
        connected[0][0][0][0] = 1
        # span:   4  4  2  5, avg = 15/4
        sp.connected_synapses[3,:] = connected.reshape(-1)

        connected = np.zeros(sp.num_inputs).reshape(sp.input_dims)
        # span:   0  0  0  0, avg = 0
        sp.connected_synapses[4,:] = connected.reshape(-1)

        true_average_connected_span = [11.0/4, 6.0/4, 14.0/4, 15.0/4, 0]

        for i in range(sp.num_minicolumns):
            connected_span = sp.average_connected_synapses_per_minicolumn(i)
            self.assertAlmostEqual(true_average_connected_span[i], connected_span)


    def test_bump_up_weak_minicolumns(self):
        sp = SpatialPooler(
            input_dims=[8], 
            minicolumn_dims=[5]
        )

        sp.synapse_perm_below_stimulus_inc = 0.01
        sp.synapse_perm_trim_threshold = 0.05
        sp.overlap_duty_cycles = np.array([0, 0.009, 0.1, 0.001, 0.002])
        sp.min_overlap_duty_cycles = np.array(5*[0.01])

        sp.potential_pools = np.array(
            [[1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]]
        )

        sp.permanences = np.array(
            [[0.200, 0.120, 0.090, 0.040, 0.000, 0.000, 0.000, 0.000],
            [0.150, 0.000, 0.000, 0.000, 0.180, 0.120, 0.000, 0.450],
            [0.000, 0.000, 0.014, 0.000, 0.032, 0.044, 0.110, 0.000],
            [0.041, 0.000, 0.000, 0.000, 0.000, 0.000, 0.178, 0.000],
            [0.100, 0.738, 0.045, 0.002, 0.050, 0.008, 0.208, 0.034]]
        )

        true_permanences = [
            [0.210, 0.130, 0.100, 0.000, 0.000, 0.000, 0.000, 0.000],
        #    Inc    Inc    Inc    Trim     -      -      -      -
            [0.160, 0.000, 0.000, 0.000, 0.190, 0.130, 0.000, 0.460],
        #    Inc      -      -       -     Inc   Inc    -      Inc
            [0.000, 0.000, 0.014, 0.000, 0.032, 0.044, 0.110, 0.000],
        #    -        -      -      -      -      -      -     -
            [0.051, 0.000, 0.000, 0.000, 0.000, 0.000, 0.188, 0.000],
        #    Inc   Trim    Trim     -      -      -    Inc      -
            [0.110, 0.748, 0.055, 0.000, 0.060, 0.000, 0.218, 0.000]
        ]

        sp.bump_up_weak_minicolumns()
        for i in range(sp.num_minicolumns):
            perm = list(sp.permanences[i,:])
            for j in range(sp.num_inputs):
                self.assertAlmostEqual(true_permanences[i][j], perm[j])


    def test_update_min_duty_cycle_local(self):
        sp = SpatialPooler(
            input_dims=(5,),
            minicolumn_dims=(8,),
            global_inhibition=False,
        )

        sp.set_inhibition_radius(1)
        sp.set_overlap_duty_cycles(np.array([0.7, 0.1, 0.5, 0.01, 0.78, 0.55, 0.1, 0.001]))
        sp.set_active_duty_cycles(np.array([0.9, 0.3, 0.5, 0.7, 0.1, 0.01, 0.08, 0.12]))
        sp.set_min_percent_overlap_duty_cycles(0.2)
        sp.update_min_duty_cycles_local()

        result_min_overlap_duty_cycles = sp.get_min_overlap_duty_cycles()
        for actual, expected in zip(result_min_overlap_duty_cycles, [0.14, 0.14, 0.1, 0.156, 0.156, 0.156, 0.11, 0.02]):
            self.assertAlmostEqual(actual, expected)


    def test_update_min_duty_cycle_global(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.num_minicolumns = 5
        sp.min_percent_overlap_duty_cycles = 0.01
        sp.overlap_duty_cycles = np.array([0.06, 1, 3, 6, 0.5])
        sp.active_duty_cycles = np.array([0.6, 0.07, 0.5, 0.4, 0.3])
        sp.update_min_duty_cycles_global()
        true_min_overlap_duty_cycles = sp.num_minicolumns*[0.01*6]
        for i in range(sp.num_minicolumns):
            self.assertAlmostEqual(true_min_overlap_duty_cycles[i], sp.min_overlap_duty_cycles[i])

        sp.min_percent_overlap_duty_cycles = 0.015
        sp.num_minicolumns = 5
        sp.overlap_duty_cycles = np.array([0.86, 2.4, 0.03, 1.6, 1.5])
        sp.active_duty_cycles = np.array([0.16, 0.007, 0.15, 0.54, 0.13])
        sp.update_min_duty_cycles_global()
        true_min_overlap_duty_cycles = sp.num_minicolumns*[0.015*2.4]
        for i in range(sp.num_minicolumns):
            self.assertAlmostEqual(true_min_overlap_duty_cycles[i], sp.min_overlap_duty_cycles[i])

        sp.min_percent_overlap_duty_cycles = 0.015
        sp.num_minicolumns = 5
        sp.overlap_duty_cycles = np.zeros(5)
        sp.active_duty_cycles = np.zeros(5)
        sp.update_min_duty_cycles_global()
        true_min_overlap_duty_cycles = sp.num_minicolumns * [0]
        for i in range(sp.num_minicolumns):
            self.assertAlmostEqual(true_min_overlap_duty_cycles[i], sp.min_overlap_duty_cycles[i])

    
    def test_adapt_synapses(self):
        sp = SpatialPooler(
            input_dims=[8],
            minicolumn_dims=[4],
            synapse_perm_dec=0.01,
            synapse_perm_inc=0.1
        )

        sp.synapse_perm_trim_threshold = 0.05

        sp.potential_pools = np.array(
            [[1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0]]
        )

        input_vector = np.array([1, 0, 0, 1, 1, 0, 1, 0])
        active_minicolumns = np.array([0, 1, 2])

        sp.permanences = np.array(
            [[0.200, 0.120, 0.090, 0.040, 0.000, 0.000, 0.000, 0.000],
            [0.150, 0.000, 0.000, 0.000, 0.180, 0.120, 0.000, 0.450],
            [0.000, 0.000, 0.014, 0.000, 0.000, 0.000, 0.110, 0.000],
            [0.040, 0.000, 0.000, 0.000, 0.000, 0.000, 0.178, 0.000]]
        )

        true_permanences = [
            [0.300, 0.110, 0.080, 0.140, 0.000, 0.000, 0.000, 0.000],
        #   Inc     Dec   Dec    Inc      -      -      -     -
            [0.250, 0.000, 0.000, 0.000, 0.280, 0.110, 0.000, 0.440],
        #   Inc      -      -     -      Inc    Dec    -     Dec
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.210, 0.000],
        #   -      -     Trim     -     -     -       Inc   -
            [0.040, 0.000, 0.000, 0.000, 0.000, 0.000, 0.178, 0.000]]
        #    -      -      -      -      -      -      -       -

        sp.adapt_synapses(input_vector, active_minicolumns)
        for i in range(sp.num_minicolumns):
            perm = list(sp.permanences[i,:])
            for j in range(sp.num_inputs):
                self.assertAlmostEqual(true_permanences[i][j], perm[j])

        sp.potential_pools = np.array(
            [[1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0]]
        )

        input_vector = np.array([1, 0, 0, 1, 1, 0, 1, 0])
        active_minicolumns = np.array([0, 1, 2])

        sp.permanences = np.array(
            [[0.200, 0.120, 0.090, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.017, 0.232, 0.400, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.014, 0.051, 0.730, 0.000, 0.000, 0.000],
            [0.170, 0.000, 0.000, 0.000, 0.000, 0.000, 0.380, 0.000]]
        )

        true_permanences = [
            [0.30, 0.110, 0.080, 0.000, 0.000, 0.000, 0.000, 0.000],
            #  Inc    Dec     Dec     -       -    -    -    -
            [0.000, 0.000, 0.222, 0.500, 0.000, 0.000, 0.000, 0.000],
            #  -     Trim    Dec    Inc    -       -      -      -
            [0.000, 0.000, 0.000, 0.151, 0.830, 0.000, 0.000, 0.000],
            #   -      -    Trim   Inc    Inc     -     -     -
            [0.170, 0.000, 0.000, 0.000, 0.000, 0.000, 0.380, 0.000]]
            #  -    -      -      -      -       -       -     -

        sp.adapt_synapses(input_vector, active_minicolumns)
        for i in range(sp.num_minicolumns):
            perm = list(sp.permanences[i,:])
            for j in range(sp.num_inputs):
                self.assertAlmostEqual(true_permanences[i][j], perm[j])

    
    def test_raise_permanence_threshold(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.input_dims=np.array([5])
        sp.minicolumn_dims=np.array([5])
        sp.synapse_perm_connected=0.1
        sp.stimulus_threshold=3
        sp.synapse_perm_below_stimulus_inc = 0.01

        sp.permanences = np.array(
            [[0.0, 0.11, 0.095, 0.092, 0.01],
            [0.12, 0.15, 0.02, 0.12, 0.09],
            [0.51, 0.081, 0.025, 0.089, 0.31],
            [0.18, 0.0601, 0.11, 0.011, 0.03],
            [0.011, 0.011, 0.011, 0.011, 0.011]]
        )

        sp.connected_synapses = np.array(
            [[0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]]
        )

        sp.connected_synapses_counts = np.array([1, 3, 2, 2, 0])

        true_permanences = [
            [0.01, 0.12, 0.105, 0.102, 0.02],  # incremented once
            [0.12, 0.15, 0.02, 0.12, 0.09],  # no change
            [0.53, 0.101, 0.045, 0.109, 0.33],  # increment twice
            [0.22, 0.1001, 0.15, 0.051, 0.07],  # increment four times
            [0.101, 0.101, 0.101, 0.101, 0.101] # increment 9 times
        ]  

        mask_potential_pools = np.array(range(5))
        for i in range(sp.num_minicolumns):
            perm = sp.permanences[i,:]
            sp.raise_permanence_to_threshold(perm, mask_potential_pools)
            for j in range(sp.num_inputs):
                self.assertAlmostEqual(true_permanences[i][j], perm[j])


    def test_update_permanences_for_minicolumn(self):
        sp = SpatialPooler(
            input_dims=[5],
            minicolumn_dims=[5],
            synapse_perm_connected=0.1,
            seed=42,
        )

        sp.synapse_perm_trim_threshold = 0.05

        permanences = np.array([
            [-0.10, 0.500, 0.400, 0.010, 0.020],
            [0.300, 0.010, 0.020, 0.120, 0.090],
            [0.070, 0.050, 1.030, 0.190, 0.060],
            [0.180, 0.090, 0.110, 0.010, 0.030],
            [0.200, 0.101, 0.050, -0.09, 1.100]],
        )

        true_connected_synapses = [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1]
        ]

        true_connected_synapses_counts = [2, 2, 2, 2, 3]

        for minicolumn_index in range(sp.num_minicolumns):
            sp.update_permanences_for_minicolumn(permanences[minicolumn_index], minicolumn_index)
            self.assertListEqual(true_connected_synapses[minicolumn_index], list(sp.connected_synapses[minicolumn_index]))

        self.assertListEqual(true_connected_synapses_counts, list(sp.connected_synapses_counts))


    def test_calculate_overlap(self):
        '''
        test that minicolumn computes overlap and percent overlap correctly.
        '''

        def calculate_overlap_percent(overlaps, connected_synapses_counts):
            return overlaps.astype(real_type) / connected_synapses_counts

        sp = SpatialPooler(
            input_dims=[10],
            minicolumn_dims=[5]
        )

        sp.connected_synapses = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
        )
        sp.connected_synapses_counts = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        input_vector = np.zeros(sp.num_inputs, dtype=real_type)
        overlaps = sp.calculate_overlap(input_vector)
        overlaps_percent = calculate_overlap_percent(overlaps, sp.connected_synapses_counts)
        true_overlaps = list(np.array([0, 0, 0, 0, 0], dtype=real_type))
        true_overlaps_percent = list(np.array([0, 0, 0, 0, 0]))
        self.assertListEqual(list(overlaps), true_overlaps)
        self.assertListEqual(list(overlaps_percent), true_overlaps_percent)


        sp.connected_synapses = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
        )
        sp.connected_synapses_counts = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        input_vector = np.ones(sp.num_inputs, dtype=real_type)
        overlaps = sp.calculate_overlap(input_vector)
        overlaps_percent = calculate_overlap_percent(overlaps, sp.connected_synapses_counts)
        true_overlaps = list(np.array([10, 8, 6, 4, 2], dtype=real_type))
        true_overlaps_percent = list(np.array([1, 1, 1, 1, 1]))
        self.assertListEqual(list(overlaps), true_overlaps)
        self.assertListEqual(list(overlaps_percent), true_overlaps_percent)


        sp.connected_synapses = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
        )
        sp.connected_synapses_counts = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        input_vector = np.zeros(sp.num_inputs, dtype=real_type)
        input_vector[9] = 1
        overlaps = sp.calculate_overlap(input_vector)
        overlaps_percent = calculate_overlap_percent(overlaps, sp.connected_synapses_counts)
        true_overlaps = list(np.array([1, 1, 1, 1, 1], dtype=real_type))
        true_overlaps_percent = list(np.array([0.1, 0.125, 1.0/6, 0.25, 0.5]))
        self.assertListEqual(list(overlaps), true_overlaps)
        self.assertListEqual(list(overlaps_percent), true_overlaps_percent)


        # zig-zag
        sp.connected_synapses = np.array(
            [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
        )
        sp.connected_synapses_counts = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        input_vector = np.zeros(sp.num_inputs, dtype=real_type)
        input_vector[range(0, 10, 2)] = 1
        overlaps = sp.calculate_overlap(input_vector)
        overlaps_percent = calculate_overlap_percent(overlaps, sp.connected_synapses_counts)
        true_overlaps = list(np.array([1, 1, 1, 1, 1], dtype=real_type))
        true_overlaps_percent = list(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
        self.assertListEqual(list(overlaps), true_overlaps)
        self.assertListEqual(list(overlaps_percent), true_overlaps_percent)


    def test_init_permanence1(self):
        '''
        test initial permanence generation. ensure that a correct amount of synapses are initialized in
        a connected state, with permanence values drawn from the correct ranges.
        '''

        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.input_dims = np.array([10])
        sp.num_inputs = 10
        sp.raise_permanence_to_threshold = Mock()


        sp.potential_radius = 2
        sp.init_connected_percent = 1
        mask = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        perm = sp.init_permanence(mask)
        connected = (perm >= sp.synapse_perm_connected).astype(int)
        num_connected = (connected.nonzero()[0]).size
        self.assertEqual(num_connected, 5)


        sp.init_connected_percent = 0
        perm = sp.init_permanence(mask)
        connected = (perm >= sp.synapse_perm_connected).astype(int)
        num_connected = (connected.nonzero()[0]).size
        self.assertEqual(num_connected, 0)


        sp.init_connected_percent = 0.5
        sp.potential_radius = 100
        sp.num_inputs = 100
        mask = np.ones(100)
        perm = sp.init_permanence(mask)
        connected = (perm >= sp.synapse_perm_connected).astype(int)
        num_connected = (connected.nonzero()[0]).size
        self.assertGreater(num_connected, 0)
        self.assertLess(num_connected, sp.num_inputs)

        min_threshold = sp.synapse_perm_min
        max_threshold = sp.synapse_perm_max
        self.assertEqual(np.logical_and((perm >= min_threshold),
                                        (perm <= max_threshold)).all(), True)


    def test_init_permanence2(self):
        '''
        test initial permanence generation. ensure that permanence values are only assigned to bits within a minicolumn's potential pool.
        '''

        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        sp.raise_permanence_to_threshold = Mock()

        sp.num_inputs = 10
        sp.init_connected_percent = 1
        mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        perm = sp.init_permanence(mask)
        connected = list((perm > 0).astype(int))
        true_connected = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertListEqual(connected, true_connected)


        sp.num_inputs = 10
        sp.init_connected_percent = 1
        mask = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        perm = sp.init_permanence(mask)
        connected = list((perm > 0).astype(int))
        true_connected = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        self.assertListEqual(connected, true_connected)


        sp.num_inputs = 10
        sp.init_connected_percent = 1
        mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        perm = sp.init_permanence(mask)
        connected = list((perm > 0).astype(int))
        true_connected = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        self.assertListEqual(connected, true_connected)


        sp.num_inputs = 10
        sp.init_connected_percent = 1
        mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1])
        perm = sp.init_permanence(mask)
        connected = list((perm > 0).astype(int))
        true_connected = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
        self.assertListEqual(connected, true_connected)


    def test_inhibit_minicolumns_global(self):
        '''
        tests that global inhibition correctly picks the correct top number of overlap scores as winning columns.
        '''
        
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        density = 0.3
        sp.num_minicolumns = 10
        overlaps = np.array([1, 2, 1, 4, 8, 3, 12, 5, 4, 1], dtype=real_type)
        active = list(sp.inhibit_minicolumns_global(overlaps, density))
        true_active = np.zeros(sp.num_minicolumns)
        true_active = [4, 6, 7]
        self.assertListEqual(list(true_active), sorted(active)) # ignore order of columns


        density = 0.5
        sp.num_minicolumns = 10
        overlaps = np.array(range(10), dtype=real_type)
        active = list(sp.inhibit_minicolumns_global(overlaps, density))
        true_active = np.zeros(sp.num_minicolumns)
        true_active = range(5, 10)
        self.assertListEqual(list(true_active), sorted(active))

    
    def test_inhibit_minicolumns_local(self):
        params = self.base_params.copy()

        sp = SpatialPooler(**params)

        density = 0.5
        sp.num_minicolumns = 10
        sp.minicolumn_dims = np.array([sp.num_minicolumns])
        sp.inhibition_radius = 2
        overlaps = np.array([1, 2, 7, 0, 3, 4, 16, 1, 1.5, 1.7], dtype=real_type)
                         #   L  W  W  L  L  W  W   L   L    W
        true_active = [1, 2, 5, 6, 9]
        active = list(sp.inhibit_minicolumns_local(overlaps, density))
        self.assertListEqual(true_active, sorted(active))


        density = 0.5
        sp.num_minicolumns = 10
        sp.minicolumn_dims = np.array([sp.num_minicolumns])
        sp.inhibition_radius = 3
        overlaps = np.array([1, 2, 7, 0, 3, 4, 16, 1, 1.5, 1.7], dtype=real_type)
                         #   L  W  W  L  W  W  W   L   L    L
        true_active = [1, 2, 4, 5, 6, 9]
        active = list(sp.inhibit_minicolumns_local(overlaps, density))
        self.assertListEqual(true_active, active)


        density = 0.3333
        sp.num_minicolumns = 10
        sp.minicolumn_dims = np.array([sp.num_minicolumns])
        sp.inhibition_radius = 3
        overlaps = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=real_type)
                         #   W  W  L  L  W  W  L  L  W  L
        true_active = [0, 1, 4, 5, 8]
        active = list(sp.inhibit_minicolumns_local(overlaps, density))
        self.assertListEqual(true_active, sorted(active))

    
    def test_RandomSpatialPoolerDoesNotLearn(self):
        sp = SpatialPooler(
            input_dims=[5],
            minicolumn_dims=[10]
        )

        input_vector = (np.random.rand(5) > 0.5).astype(uint_type)
        active_array = np.zeros(sp.num_minicolumns).astype(real_type)

        # should start off at 0
        self.assertEqual(sp.iteration_num, 0)
        self.assertEqual(sp.iteration_learn_num, 0)

        # store the initialized state
        initial_permanences = copy(sp.permanences)

        sp.compute(input_vector, False, active_array)
        # should have incremented general counter but not learning counter
        self.assertEqual(sp.iteration_num, 1)
        self.assertEqual(sp.iteration_learn_num, 0)

        # check the initial perm state was not modified either
        self.assertEqual((sp.permanences == initial_permanences).all(), True)


if __name__ == '__main__':
    unittest.main()