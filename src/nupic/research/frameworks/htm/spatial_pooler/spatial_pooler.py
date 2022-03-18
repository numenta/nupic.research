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

import itertools

import numpy as np

real_type = np.float32
uint_type = np.uint32

class SpatialPooler:
    """
    The HTM Spatial Pooler (SP) models how neurons learn feedforward connections and form
    efficient representations of the input. Converts arbitrary binary input patterns into
    sparse distributed representations (SDRs) using competitive Hebbian learning and
    homeostatic excitability control (boosting).

    For more information, refer to the paper:
    https://www.frontiersin.org/articles/10.3389/fncom.2017.00111/full
    """

    def __init__(
        self,
        input_dims=(32, 32),
        minicolumn_dims=(64, 64),
        active_minicolumns_per_inh_area=10,
        local_density=-1.0,
        potential_radius=16,
        potential_percent=0.5,
        global_inhibition=False,
        stimulus_threshold=0,
        synapse_perm_inc=0.05,
        synapse_perm_dec=0.008,
        synapse_perm_connected=0.1,
        min_percent_overlap_duty_cycles=0.001,
        duty_cycle_period=1000,
        boost_strength=0.0,
        seed=-1,
    ):
        """
        input_dims:                         dimensions of input vector.
                                            default ``(32, 32)``.

        minicolumn_dims:                    dimensions of cortical minicolumns.
                                            default ``(64, 64)``.

        active_minicolumns_per_inh_area:    sets number of minicolumns that remain
                                            on within a local inhibition area. as
                                            minicolumns learn and grow their effective
                                            receptive fields, the inhibition radius
                                            will also grow. when this happens, net
                                            density of the number of active minicolumns
                                            *decreases* if this method is used.
                                            default ``10.0``.

        local_density:                      sets desired density of active minicolumns
                                            within local inhibition area. ensures that
                                            at most N minicolumns remain on within a
                                            local inhibition area, where

                                            N = local_density * (# of minicolumns in
                                            inhibition area).

                                            density of active minicolumns stays the same
                                            regardless of size of minicolumns' receptive
                                            fields.
                                            default ``-1.0``.

        potential_radius:                   number of input bits visible to each
                                            minicolumn. large enough value means that
                                            every minicolumn can potentially connect to
                                            every input bit. this parameter defines a
                                            square/hypercube area: a minicolumn will
                                            have a max square potential pool with side
                                            lengths:

                                            (2 * potential_radius + 1).

                                            default ``16``.

        potential_percent:                  percent of inputs within a minicolumn's
                                            potential_radius that a minicolumn can be
                                            connected to. if 1, minicolumn will be
                                            connected to every input within its
                                            potential radius. at initialization, we
                                            choose

                                            ((2*potential_radius + 1)^(# input dims)) *
                                            potential_percent

                                            # of input bits to comprise the minicolumn's
                                            potential pool.
                                            default ``0.5``.

        global_inhibition:                  if True, then during inhibition the winning
                                            minicolumns are selected as the most active
                                            minicolumns from the region as a whole.
                                            else, winning minicolumns are selected
                                            w.r.t. their local neighborhoods.
                                            default ``False``.

        stimulus_threshold:                 minimum number of synapses that must be on
                                            in order for a minicolumn to turn on.
                                            prevents noise from activating minicolumns.
                                            specified as a percent of a fully grown
                                            synapse.
                                            default ``0``.

        synapse_perm_inc:                   amount by which an active synapse is
                                            incremented in each round. specified as a
                                            percent of a fully grown synapse.
                                            default ``0.05``.

        synapse_perm_dec:                   amount by which an inactive synapse is
                                            decremented in each round. specified as a
                                            percent of a fully grown synapse.
                                            default ``0.008``.

        synapse_perm_connected:             default connected threshold. any synapse
                                            whose permanence value is above the
                                            connected threshold is "connected", meaning
                                            it can conribute to the cell's firing.
                                            default ``0.1``.

        min_percent_overlap_duty_cycles:    floor on how often a minicolumn should have
                                            at least stimulus_threshold active inputs.
                                            periodically, each minicolumn looks at the
                                            overlap duty cycle of all other minicolumns
                                            within its inhibition radius and sets its
                                            own internal minimal acceptable duty cycle
                                            to

                                            (minimum % duty cycle before inhibition) *
                                            max(other minicolumns' duty cycles).

                                            if any minicolumn whose overlap duty cycle
                                            falls below this, all permanence values get
                                            boosted by synapse_perm_inc. raising
                                            permanences before inhibition allows a cell
                                            to search for new inputs when either its
                                            previously learned inputs are no longer ever
                                            active or when vast majority of inputs are
                                            "hijacked" by other minicolumns.
                                            default ``0.001``.

        duty_cycle_period:                  period used to calculate duty cycles. higher
                                            values means it takes longer to respond in
                                            changes in boosting or synapses per
                                            connected cell. shorter values make it more
                                            unstable and likely to oscillate.
                                            default ``1000``.

        boost_strength:                     a number >= 0.0 that is used to control the
                                            strength of boosting. boosting increases as
                                            a function of boost_strength. encourages
                                            minicolumns to have similar active duty
                                            cycles as their neighbors, which will lead
                                            to more efficient use of minicolumns. too
                                            much boosting may lead to instability of
                                            spatial pooler outputs.
                                            default ``0.0``.

        seed:                               seed for numpy random generator.
        """

        # controls # of minicolumns that are on within a local inhibition area
        self.active_minicolumns_per_inh_area = int(active_minicolumns_per_inh_area)
        self.local_density = local_density

        assert (self.active_minicolumns_per_inh_area > 0) or \
               (0 < self.local_density <= 0.5)

        # input and and minicolumn dimensionality
        self.input_dims = np.array(input_dims, ndmin=1)
        self.minicolumn_dims = np.array(minicolumn_dims, ndmin=1)
        self.num_inputs = np.prod(self.input_dims)
        self.num_minicolumns = np.prod(self.minicolumn_dims)

        assert (self.num_inputs > 0) and (self.num_minicolumns > 0)
        assert self.input_dims.size == self.minicolumn_dims.size

        # global or local inhibition
        self.global_inhibition = global_inhibition

        assert isinstance(self.global_inhibition, bool)

        # defines extent of input that each minicolumn can potentially connect to
        self.potential_percent = potential_percent
        self.potential_radius = int(min(potential_radius, self.num_inputs))

        # rules for synaptic connections
        self.stimulus_threshold = stimulus_threshold
        self.synapse_perm_inc = synapse_perm_inc
        self.synapse_perm_dec = synapse_perm_dec
        self.synapse_perm_below_stimulus_inc = synapse_perm_connected / 10.0
        self.synapse_perm_connected = synapse_perm_connected
        self.synapse_perm_min = 0.0
        self.synapse_perm_max = 1.0
        self.synapse_perm_trim_threshold = synapse_perm_inc / 2.0
        self.permanence_epsilon = 0.000001

        assert self.synapse_perm_trim_threshold < self.synapse_perm_connected

        # cycle period, update period, and iteration count
        self.duty_cycle_period = duty_cycle_period
        self.update_period = 50
        self.iteration_num = 0
        self.iteration_learn_num = 0

        # duty cycles + overlap metrics + boosting
        #
        # - overlap_duty_cycles[i] shows the moving average of the number of inputs
        #   which overlaps with minicolumn "i".
        #
        # - active_duty_cycles[i] shows the moving average of the frequency of
        #   activation for minicolumn "i".
        #
        # - min_overlap_duty_cycles[i] is minimum duty cycles defining normal activity
        #   for minicolumn "i". a minicolumn with active_duty_cycle below this threshold
        #   is boosted.
        #
        # - boost_factors[i] is the boost factor for minicolumn "i". used to increase
        #   the overlap of inactive minicolumns to improve their chances of becoming
        #   active. depends on active_duty_cycles via an exponential function.
        #
        # - overlaps[i] determines overlap between minicolumn "i" and input vector.
        #   overlap = number of connected synapses to input bits which are on.
        #
        # - boosted_overlaps[i] = boost_factors[i] * overlaps[i] if spatial pooler is
        #   learning, else boosted_overlaps[i] = overlaps[i]
        self.overlap_duty_cycles = np.zeros(self.num_minicolumns, dtype=real_type)
        self.active_duty_cycles = np.zeros(self.num_minicolumns, dtype=real_type)
        self.min_overlap_duty_cycles = np.zeros(self.num_minicolumns, dtype=real_type)
        self.min_percent_overlap_duty_cycles = min_percent_overlap_duty_cycles
        self.init_connected_percent = 0.5
        self.boost_strength = boost_strength
        self.boost_factors = np.ones(self.num_minicolumns, dtype=real_type)
        self.overlaps = np.zeros(self.num_minicolumns, dtype=real_type)
        self.boosted_overlaps = np.zeros(self.num_minicolumns, dtype=real_type)

        # random seed
        if seed == -1:
            self.seed = np.random.randint(1e10)
        else:
            self.seed = seed

        self.generator = np.random.default_rng(seed=self.seed)

        # rows of matrix represent minicolumns: i.e. map from input bits to minicolumns.
        # columns of matrix represent input bits: i.e. map from minicolumns to input
        # bits.
        #
        # - potential_pools[i][j] shows if input bit "j" is in the potential pool of
        #   minicolumn "i". a minicolumn can only be connected to inputs in its
        #   potential pool.
        #
        # - permanences[i][j] shows the permanence for minicolumn "i" to input bit "j".
        #
        # - connected_synapses[i][j] shows if minicolumn "i" is connnected to
        #   input bit "j".
        #
        # - connected_synapse_counts[i] shows the number of connected synapses for
        #   minicolumn "i".
        self.potential_pools = np.zeros((self.num_minicolumns, self.num_inputs),
                                        dtype=np.bool_)

        self.permanences = np.zeros((self.num_minicolumns, self.num_inputs),
                                    dtype=real_type)

        self.connected_synapses = np.zeros((self.num_minicolumns, self.num_inputs),
                                           dtype=np.bool_)

        self.connected_synapses_counts = np.zeros(self.num_minicolumns, dtype=real_type)

        for minicolumn_index in range(self.num_minicolumns):
            potential = self.map_potential(minicolumn_index)
            self.potential_pools[minicolumn_index, potential.nonzero()[0]] = 1
            permanence = self.init_permanence(potential)
            self.update_permanences_for_minicolumn(permanence, minicolumn_index,
                                                   raise_perm=True)

        self.inhibition_radius = 0
        self.update_inhibition_radius()

    def compute(self, input_vector, learn, active_array):
        """
        primary spatial pooler method. takes an input vector and outputs the indices of
        the active minicolumns. if learn is True, update the permanences of the
        minicolumns.

        input vector is a binary np.array that comprises the input to the spatial
        pooler. array is treated as a 1D array. only requirement is that the number of
        bits in input vector must match the number of bits specified by the call to the
        constructor. i.e., there must be a 0 or 1 in the array for every input bit.

        active_array is an array whose size is equal to the number of minicolumns. will
        be populated with 1's at the indices of the active minicolumns and 0's
        everywhere else.
        """

        assert isinstance(input_vector, np.ndarray) and \
            (input_vector.size == self.num_inputs)

        self.iteration_num += 1
        if learn:
            self.iteration_learn_num += 1

        input_vector = np.array(input_vector, dtype=real_type).reshape(-1)

        self.overlaps = self.calculate_overlap(input_vector)

        # apply boosting when learning is on
        if learn:
            self.boosted_overlaps = self.boost_factors * self.overlaps
        else:
            self.boosted_overlaps = self.overlaps

        # apply inhibition to determine the winning minicolumns
        active_minicolumns = self.inhibit_minicolumns(self.boosted_overlaps)

        if learn:
            self.adapt_synapses(input_vector, active_minicolumns)
            self.update_duty_cycles(self.overlaps, active_minicolumns)
            self.bump_up_weak_minicolumns()
            self.update_boost_factors()

            if (self.iteration_num % self.update_period) == 0:
                self.update_inhibition_radius()
                self.update_min_duty_cycles()

        active_array.fill(0)
        active_array[active_minicolumns] = 1

    def map_potential(self, index):
        """
        maps minicolumn to input bits.

        takes index of minicolumn and determines what indices of the input vector are
        located within the minicolumn's potential pool. returns list containing indices
        of the input bits.

        if the potential radius is greater than or equal to the largest input dimension,
        then each minicolumn connects to all of the inputs.
        """
        center_input = self.map_minicolumn(index)
        minicolumn_inputs = self.get_input_neighborhood(center_input).astype(uint_type)

        # select a subset of the receptive field to serve as the potential pool
        num_potential = int(minicolumn_inputs.size * self.potential_percent + 0.5)

        selected_inputs = self.generator.choice(minicolumn_inputs, size=num_potential,
                                                replace=False).astype(uint_type)

        potential = np.zeros(self.num_inputs, dtype=uint_type)
        potential[selected_inputs] = 1

        return potential

    def map_minicolumn(self, index):
        """
        maps a minicolumn index to respective input index, keeping topology of the
        region.

        in other words, takes index of minicolumn as argument and calculates index of
        flattened input vector that is to be the center of the minicolumn's potential
        pool.
        """

        minicolumn_coordinates = np.array(np.unravel_index(index, self.minicolumn_dims),
                                          dtype=real_type)

        # map minicolumn index to find appropriate input index
        input_coordinates = self.input_dims * \
            (minicolumn_coordinates / self.minicolumn_dims)

        # shift input index by (1/2M)(I)
        input_coordinates += (0.5 * self.input_dims) / self.minicolumn_dims

        # get valid index
        input_coordinates = input_coordinates.astype(int)

        input_index = np.ravel_multi_index(input_coordinates, self.input_dims)

        return input_index

    def get_input_neighborhood(self, center_input):
        """
        return a neighborhood of inputs.
        """

        return self.neighborhood(center_input, self.potential_radius, self.input_dims)

    def get_minicolumn_neighborhood(self, center_minicolumn):
        """
        return a neighborhood of minicolumns.
        """

        return self.neighborhood(center_minicolumn, self.inhibition_radius,
                                 self.minicolumn_dims)

    def neighborhood(self, center_input, radius, dimensions):
        """
        gets the points in the neighborhood of a point. a point's neighborhood is the
        n-dimensional hypercube with sides ranging [center - radius, center + radius]
        inclusive. if there are two dimensions and radius = 3, then neighborhood is 6x6.
        neighborhoods are truncated when they are near an edge.
        """

        center_position = np.unravel_index(center_input, dimensions)

        intervals = []
        for i, dimension in enumerate(dimensions):
            left = max(0, center_position[i] - radius)
            right = min(dimension - 1, center_position[i] + radius)
            intervals.append(range(left, right + 1))

        coordinates = np.array(list(itertools.product(*intervals)))

        return np.ravel_multi_index(coordinates.T, dimensions)

    def update_inhibition_radius(self):
        """
        update inhibition radius, which is a measure of the hypersquare of minicolumns
        that each minicolumn is "connected to" on average. since minicolumns are not
        connected to each other directly, we determine this quantity by figuring out how
        many *inputs* a minicolumn is connected to and multiply this by the total number
        of minicolumns that exist for each input. for multiple dimensions, these
        calculations are averaged over all dimensions of inputs and minicolumns.

        this computation is not used if global_inhibition is enabled.
        """

        if self.global_inhibition:
            self.inhibition_radius = int(self.minicolumn_dims.max())
            return

        # how many inputs a minicolumn is connected to on average
        average_connected_span = np.average(
            [
                self.average_connected_synapses_per_minicolumn(m)
                for m in range(self.num_minicolumns)
            ]
        )

        # how many minicolumns exist for each input on average
        minicolumns_per_input = self.average_minicolumns_per_input()

        diameter = average_connected_span * minicolumns_per_input
        radius = (diameter - 1) / 2.0
        radius = max(radius, 1.0)

        self.inhibition_radius = int(radius + 0.5)

    def average_connected_synapses_per_minicolumn(self, minicolumn_index):
        """
        range of connected synapses per minicolumn, averaged for each dimension. value
        is used to calculate the inhibition radius.
        """

        connected = self.connected_synapses[minicolumn_index, :].nonzero()[0]

        if connected.size == 0:
            return 0
        else:
            min_coordinate = np.empty(self.input_dims.size)
            max_coordinate = np.empty(self.input_dims.size)

            min_coordinate.fill(max(self.input_dims))
            max_coordinate.fill(-1)

            for i in connected:
                min_coordinate = np.minimum(min_coordinate,
                                            np.unravel_index(i, self.input_dims))
                max_coordinate = np.maximum(max_coordinate,
                                            np.unravel_index(i, self.input_dims))

            return np.average(max_coordinate - min_coordinate + 1)

    def average_minicolumns_per_input(self):
        """
        average number of minicolumns per input, value is used to calculate the
        inhibition radius.

        if the minicolumn dimensions don't match the input dimensions, missing
        dimensions are treated as "ones".
        """

        num_dim = max(self.minicolumn_dims.size, self.input_dims.size)

        minicol_dim = np.ones(num_dim)
        minicol_dim[: self.minicolumn_dims.size] = self.minicolumn_dims

        input_dim = np.ones(num_dim)
        input_dim[: self.input_dims.size] = self.input_dims

        minicolumns_per_input = minicol_dim.astype(real_type) / input_dim

        return np.average(minicolumns_per_input)

    def init_permanence(self, potential):
        """
        takes in a np.array specifying the potential pool of the minicolumn.
        initializes the permanence of a minicolumn. returns a 1D array the size of the
        input, where each entry represents the initial permanence value between the
        input bit at the index and minicolumn represented by the index.
        """

        permanence = np.zeros(self.num_inputs, dtype=real_type)

        for i in range(self.num_inputs):
            if potential[i] < 1:
                continue

            if self.generator.random() <= self.init_connected_percent:
                # initialize connected permanence by using a a randomly generated
                # permanence value that is close to synapse_perm_connected

                p = (
                    self.synapse_perm_connected
                    + (self.synapse_perm_max - self.synapse_perm_connected)
                    * self.generator.random()
                )
                p = int(p * 100000) / 100000.0

                permanence[i] = p
            else:
                # initialize unconnected permanence

                p = self.synapse_perm_connected * self.generator.random()
                p = int(p * 100000) / 100000.0

                permanence[i] = p

        # clip off low values
        permanence[permanence < self.synapse_perm_trim_threshold] = 0

        return permanence

    def update_permanences_for_minicolumn(self, permanence, minicolumn_index,
                                          raise_perm=True):
        """
        updates the permanence matrix with a minicolumn's new permanence values.

        this method is responsible for clipping the permanence values so they remain
        between 0 and 1. also responsible for trimming permanence values below
        synapse_perm_trim_threshold, which enforces sparsity.

        permanences is a dense array that describes permanence values for each
        minicolumn (dense = contains all the zeros and non-zero values).
        raise_perm is a boolean value indicating whether permanence values should be
        raised until a minimum number of synapses are in a connected state.
        should be False when a direct assignment is required.
        """

        mask_potential = np.where(self.potential_pools[minicolumn_index, :] > 0)[0]

        if raise_perm:
            self.raise_permanence_to_threshold(permanence, mask_potential)

        permanence[permanence < self.synapse_perm_trim_threshold] = 0

        np.clip(permanence, self.synapse_perm_min, self.synapse_perm_max,
                out=permanence)
        self.permanences[minicolumn_index, :] = permanence

        new_connected = np.where(
            permanence >= (self.synapse_perm_connected - self.permanence_epsilon)
        )[0]

        # remove old synaptic connections and make new ones
        self.connected_synapses[minicolumn_index, :] = 0
        self.connected_synapses[minicolumn_index, new_connected] = 1
        self.connected_synapses_counts[minicolumn_index] = new_connected.size

    def raise_permanence_to_threshold(self, permanence, mask_potential):
        """
        ensures that each minicolumn has enough connections to input bits to allow it to
        become active. takes in an array of permanence values for a minicolumn and its
        synapse indices whose permanences need to be raised.

        reasoning: since a minicolumn must have at least stimulus_threshold overlaps in
        order to be considered during the inhibition phase, minicolumns without this
        minimal number of connections have no chance of reaching the threshold, even if
        all input bits they are connected to are on. for such minicolumns, the
        permanence values are increased until the minimum number of connections are
        formed.
        """

        assert len(mask_potential) >= self.stimulus_threshold

        np.clip(permanence, self.synapse_perm_min, self.synapse_perm_max,
                out=permanence)

        while True:
            num_connected = np.nonzero(
                permanence > (self.synapse_perm_connected - self.permanence_epsilon)
            )[0].size

            if num_connected >= self.stimulus_threshold:
                return
            else:
                permanence[mask_potential] += self.synapse_perm_below_stimulus_inc

    def calculate_overlap(self, input_vector):
        """
        determines each minicolumn's overlap with the current input vector. overlap of a
        minicolumn is the # of connected synapses to input bits which are turned on.
        """

        overlaps = np.zeros(self.num_minicolumns, dtype=real_type)

        for m in range(self.num_minicolumns):
            overlaps[m] = input_vector[self.connected_synapses[m, :] > 0].sum()

        return overlaps

    def inhibit_minicolumns(self, overlaps):
        """
        performs inhibition. calculates the necessary values needed to actually perform
        inhibition (either global or local inhibition).

        takes in overlaps, which is an array containing overlap score for each
        minicolumn. overlap score for a minicolumn = number of connected synapses to
        input bits which are turned on.
        """

        if self.local_density > 0:
            density = self.local_density
        else:
            inhibition_area = min(
                self.num_minicolumns,
                (2 * self.inhibition_radius + 1) ** self.minicolumn_dims.size,
            )

            density = float(self.active_minicolumns_per_inh_area) / inhibition_area
            density = min(density, 0.5)

        if self.global_inhibition or self.inhibition_radius > max(self.minicolumn_dims):
            return self.inhibit_minicolumns_global(overlaps, density)
        else:
            return self.inhibit_minicolumns_local(overlaps, density)

    def inhibit_minicolumns_global(self, overlaps, density):
        """
        global inhibition -- pick the top num_active minicolumns with the highest
        overlap score in the entire region.

        at most half of the minicolumns in a local neighborhood are allowed to be
        active. minicolumns with an overlap score below the stimulus_threshold are
        always inhibited.
        """

        # calculate num_active minicolumns per inhibition area
        num_active = int(density * self.num_minicolumns)

        # calculate winners using sorting algorithm
        winning_minicolumn_indices = np.argsort(overlaps, kind="mergesort")

        # enforce the stimulus threshold
        start = len(winning_minicolumn_indices) - num_active
        while start < len(winning_minicolumn_indices):
            if overlaps[winning_minicolumn_indices[start]] >= self.stimulus_threshold:
                # once one minicolumn has an overlap greater than the threshold, the
                # rest in the sorted list will also be higher than the threshold
                break
            else:
                start += 1

        return winning_minicolumn_indices[start:][::-1]

    def inhibit_minicolumns_local(self, overlaps, density):
        """
        local inhibition -- performed on a minicolumn by minicolumn basis. each
        minicolumn observes the overlaps of its neighbors and is selected if its overlap
        score is within the top num_active in its local neighborhood.

        at most half of minicolumns in a local neighborhood are allowed to be active.
        minicolumns with an overlap score below the stimulus_threshold are always
        inhibited.
        """

        active_array = np.zeros(self.num_minicolumns, dtype=np.bool_)

        for minicolumn, overlap in enumerate(overlaps):
            if overlap >= self.stimulus_threshold:
                neighborhood = self.get_minicolumn_neighborhood(minicolumn)

                neighborhood_overlaps = overlaps[neighborhood]

                # # of neighbors with overlap value greater than current minicolumn
                num_bigger = np.count_nonzero(neighborhood_overlaps > overlap)

                # when there is a tie (neighboring minicolumns have same overlap as
                # current minicolumn), favor neighbors already selected as active
                tied_neighbors = neighborhood[
                    np.where(neighborhood_overlaps == overlap)
                ]
                num_ties_lost = np.count_nonzero(active_array[tied_neighbors])

                # maximum number of active minicolumns in neighborhood
                num_active = int(0.5 + density * len(neighborhood))

                # activate current minicolumn if enough neighbors aren't "better"
                # (better = higher overlap or same overlap value but already active)
                if (num_bigger + num_ties_lost) < num_active:
                    active_array[minicolumn] = True

        return active_array.nonzero()[0]

    def adapt_synapses(self, input_vector, active_minicolumns):
        """
        primary method in charge of learning. adapts the permanence values of the
        synapses based on input vectors, and the chosen minicolumns after inhibition
        round.

        permanence values are increased for synapses connected to ON input bits.
        permanence values are decreased for synapses connected to OFF input bits.
        """

        input_indices = np.where(input_vector > 0)[0]

        # for input bits that are on, increase synaptic permanence. for inputs bits that
        # are off, decrease synaptic permanence.
        permanence_changes = np.zeros(self.num_inputs, dtype=real_type)
        permanence_changes.fill(-1 * self.synapse_perm_dec)
        permanence_changes[input_indices] = self.synapse_perm_inc

        for minicolumn_index in active_minicolumns:
            # find synaptic permanences for all input bits of current minicolumn
            permanence = self.permanences[minicolumn_index, :]

            # find which input bits can *possibly* connect to current minicolumn
            mask_potential = np.where(self.potential_pools[minicolumn_index, :] > 0)[0]

            # only update synaptic permanences for input bits that can *possibly*
            # connect to current minicolumn
            permanence[mask_potential] += permanence_changes[mask_potential]

            # do the update
            self.update_permanences_for_minicolumn(permanence, minicolumn_index,
                                                   raise_perm=True)

    def update_duty_cycles(self, overlaps, active_minicolumns):
        """
        updates the duty cycles for each minicolumn.

        overlap duty cycles is a moving average of the number of inputs which overlapped
        with each minicolumn. active duty cycles is a moving average of the frequency of
        activation for each minicolumn.

                        (period - 1)*duty_cycle + new_value
        duty_cycle := ---------------------------------------
                                period
        """

        period = self.duty_cycle_period
        if period > self.iteration_num:
            period = self.iteration_num
        assert period >= 1

        overlap_array = np.zeros(self.num_minicolumns, dtype=real_type)
        overlap_array[overlaps > 0] = 1
        self.overlap_duty_cycles = (
            self.overlap_duty_cycles * (period - 1.0) + overlap_array
        ) / period

        active_array = np.zeros(self.num_minicolumns, dtype=real_type)
        active_array[active_minicolumns] = 1
        self.active_duty_cycles = (
            self.active_duty_cycles * (period - 1.0) + active_array
        ) / period

    def bump_up_weak_minicolumns(self):
        """
        increases the permanence values of synapses of minicolumns whose activity level
        has been too low. such minicolumns are identified by having an overlap duty
        cycle that drops too much below those of their peers. the permanence values for
        such minicolumns are increased.
        """

        weak_minicolumns = np.where(
            self.overlap_duty_cycles < self.min_overlap_duty_cycles
        )[0]

        for minicolumn_index in weak_minicolumns:
            # find synaptic permanences for all input bits of current minicolumn
            permanence = self.permanences[minicolumn_index, :].astype(real_type)

            # find which input bits can *possibly* connect to current minicolumn
            mask_potential = np.where(self.potential_pools[minicolumn_index, :] > 0)[0]

            # only update synaptic permanences for input bits that can *possibly*
            # connect to current minicolumn
            permanence[mask_potential] += self.synapse_perm_below_stimulus_inc

            # do the update
            self.update_permanences_for_minicolumn(permanence, minicolumn_index,
                                                   raise_perm=False)

    def update_boost_factors(self):
        """
        update boost_factors for all minicolumns. boost_factors are used to increase the
        overlap of inactive minicolumns to improve their chances of becoming active.

        boost_factors = exp[ - boost_strength * (duty_cycle - target_density) ]

        minicolumns that have been active at the target activation level -->
            boost_factors of 1 (their overlap is not boosted)
        minicolumns that have an active duty cycle below their neighbors -->
            boosted depending on how infrequently they have been active
        minicolumns that have an active duty cycle above the target activation level -->
            boost_factors below 1 (their overlap is suppressed)

        boost_factors depends on the active_duty_cycle via an exponential function

        boost_factors
                ^
                |
                |\
                | \
          1  _  |  \
                |    _
                |      _ _
                |          _ _ _ _
                +--------------------> activeDutyCycle
                   |
              targetDensity
        """

        if self.global_inhibition:
            self.update_boost_factors_global()
        else:
            self.update_boost_factors_local()

    def update_boost_factors_global(self):
        """
        update boost factors when global inhibition is used. target_density is the
        sparsity of the spatial pooler.
        """

        if self.local_density > 0:
            target_density = self.local_density
        else:
            inhibition_area = min(
                (2 * self.inhibition_radius + 1) ** self.minicolumn_dims.size,
                self.num_minicolumns,
            )

            target_density = min(
                float(self.active_minicolumns_per_inh_area) / inhibition_area,
                0.5
            )

        self.boost_factors = np.exp(
            -self.boost_strength * (self.active_duty_cycles - target_density)
        )

    def update_boost_factors_local(self):
        """
        update boost factors when local inhibition is used. target_density is the
        average active_duty_cycles of the neighboring minicolumns of each minicolumn.
        """

        target_density = np.zeros(self.num_minicolumns, dtype=real_type)

        for m in range(self.num_minicolumns):
            mask_neighbors = self.get_minicolumn_neighborhood(m)

            target_density[m] = np.mean(self.active_duty_cycles[mask_neighbors])

        self.boost_factors = np.exp(
            -self.boost_strength * (self.active_duty_cycles - target_density)
        )

    def update_min_duty_cycles(self):
        """
        updates minimum duty cycles defining normal activity for a minicolumn. a
        minicolumn with active_duty_cycle below this threshold is boosted.
        """

        if self.global_inhibition or self.inhibition_radius > max(self.minicolumn_dims):
            self.update_min_duty_cycles_global()
        else:
            self.update_min_duty_cycles_local()

    def update_min_duty_cycles_global(self):
        """
        set the minimum duty cycles for the overlap of all minicolumns to be a percent
        of the maximum in the region, specified by min_percent_overlap_duty_cycles.
        """

        self.min_overlap_duty_cycles.fill(
            self.min_percent_overlap_duty_cycles * self.overlap_duty_cycles.max()
        )

    def update_min_duty_cycles_local(self):
        """
        each minicolumn's duty cycles are set to be a percent of the maximum duty cycles
        in the minicolumn's neighborhood.
        """

        for minicolumn in range(self.num_minicolumns):
            neighborhood = self.get_minicolumn_neighborhood(minicolumn)

            max_overlap_duty = self.overlap_duty_cycles[neighborhood].max()

            self.min_overlap_duty_cycles[minicolumn] = (
                self.min_percent_overlap_duty_cycles * max_overlap_duty
            )

    # getter methods

    def get_num_inputs(self):
        return self.num_inputs

    def get_num_minicolumns(self):
        return self.num_minicolumns

    def get_iteration_learn_num(self):
        return self.iteration_learn_num

    def get_boost_factors(self):
        return self.boost_factors

    def get_active_duty_cycles(self):
        return self.active_duty_cycles

    def get_potential_pools(self):
        return self.potential_pools

    def get_permanences(self):
        return self.permanences

    def get_connected_synapses(self):
        return self.connected_synapses

    def get_connected_synapses_counts(self):
        return self.connected_synapses_counts

    def get_overlaps(self):
        return self.overlaps

    def get_boosted_overlaps(self):
        return self.boosted_overlaps

    def get_min_overlap_duty_cycles(self):
        return self.min_overlap_duty_cycles

    # setter methods

    def set_inhibition_radius(self, radius):
        self.inhibition_radius = radius

    def set_boost_factors(self, boost_factors):
        self.boost_factors = boost_factors

    def set_overlap_duty_cycles(self, overlap_duty_cycles):
        self.overlap_duty_cycles = overlap_duty_cycles

    def set_active_duty_cycles(self, active_duty_cycles):
        self.active_duty_cycles = active_duty_cycles

    def set_min_percent_overlap_duty_cycles(self, min_percent_overlap_duty_cycles):
        self.min_percent_overlap_duty_cycles = min_percent_overlap_duty_cycles
