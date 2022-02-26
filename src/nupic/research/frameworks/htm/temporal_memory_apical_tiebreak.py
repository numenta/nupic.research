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

from collections import defaultdict
from enum import unique

import torch

real_type = torch.float32
int_type = torch.int32

class TemporalMemoryApicalTiebreak():
    """
    generalized Temporal Memory (TM) with apical dendrites that add a "tiebreak".

    basal connections implement traditional TM. apical connections used for further
    disambiguation.

    if multiple cells in a minicolumn have active basal segments,
    each of those cells is predicted unless one has an active apical segment. if so,
    only the cells with active basal and apical segments are predicted.

    in other words, apical connections have no effect unless basal input is a union
    of SDRs (e.g. from bursting minicolumns).

    the class is generalized in two ways:
        - exposes two main methods `depolarize_cells` and `activate_cells`.
        - TM is unaware of whether `basal_input` or `apical_input` are from
          internal/external cells -- caller knows what these cells numbers mean
        - TM does not specify when a "timestep" begins and ends. callers or subclasses
          can introduce the notion of a timestep.
    """

    def __init__(
        self,
        num_minicolumns=2048,
        basal_input_size=0,
        apical_input_size=0,
        num_cells_per_minicolumn=32,
        activation_threshold=13,
        reduced_basal_threshold=13,
        initial_permanence=0.21,
        connected_permanence=0.50,
        matching_threshold=10,
        sample_size=20,
        permanence_increment=0.1,
        permanence_decrement=0.1,
        basal_predicted_segment_decrement=0.0,
        apical_predicted_segment_decrement=0.0,
        max_synapses_per_segment=-1,
        seed=-1
    ):
        """
        num_minicolumns:                            number of minicolumns.
                                                    default `2048`.

        basal_input_size:                           number of bits in the basal input.
                                                    default `0`.

        apical_input_size:                          number of bits in the apical input.
                                                    default `0`.

        num_cells_per_minicolumn:                   number of cells per minicolumn.
                                                    default `32`.

        activation_threshold:                       if # of active connected synapses
                                                    on a segment is at least this
                                                    threshold, the segment is "active".
                                                    default `13`.

        reduced_basal_threshold:                    activation threshold of basal
                                                    (lateral) segments for cells that
                                                    have active apical segments. if
                                                    equal to activation_threshold,
                                                    this parameter has no effect.
                                                    default `13`.

        initial_permanence:                         initial permanence of a new synapse.
                                                    default `0.21`.

        connected_permanence:                       if the permanence value for a
                                                    synapse is greater than this vaulue,
                                                    it is connected.
                                                    default `0.5`.

        matching_threshold:                         if # of potential synapses active on
                                                    a segment is at least this
                                                    threshold, then synapses are
                                                    "matching" and eligible for
                                                    learning.
                                                    default `10`.

        sample_size:                                how much of the active SDR to sample
                                                    with synapses.
                                                    default `20`.

        permanence_increment:                       amount by which permanences of
                                                    synapses are incremented during
                                                    learning.
                                                    default `0.1`.

        permanence_decrement:                       amount by which permanences of
                                                    synapses are decremented during
                                                    learning.
                                                    default `0.1`.

        basal_predicted_segment_decrement:          amount by which permanences are
                                                    punished for incorrect predictions.
                                                    default `0.0`.

        apical_predicted_segment_decrement:         amount by which segments are
                                                    punished for incorrect predictions.
                                                    default `0.0`.

        max_synapses_per_segment:                   max number of synapses per segment.
                                                    default `-1`.

        seed:                                       seed for random number generator.
                                                    default `-1`.
        """

        self.num_minicolumns = num_minicolumns
        self.basal_input_size = basal_input_size
        self.apical_input_size = apical_input_size
        self.num_cells_per_minicolumn = num_cells_per_minicolumn
        self.activation_threshold = activation_threshold
        self.reduced_basal_threshold = reduced_basal_threshold
        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence
        self.matching_threshold = matching_threshold
        self.sample_size = sample_size
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.basal_predicted_segment_decrement = basal_predicted_segment_decrement
        self.apical_predicted_segment_decrement = apical_predicted_segment_decrement
        self.max_synapses_per_segment = max_synapses_per_segment

        # random seed
        if seed == -1:
            self.seed = torch.random.seed()
        else:
            self.seed = seed

        self.generator = torch.random.manual_seed(self.seed)

        self.basal_connections = torch.zeros(
            self.num_minicolumns * self.num_cells_per_minicolumn,
            self.basal_input_size,
            dtype=int_type
        )

        self.apical_connections = torch.zeros(
            self.num_minicolumns * self.num_cells_per_minicolumn,
            self.apical_input_size,
            dtype=int_type
        )


        #
        #
        #
        #
        #
        #
        #
        self.active_cells = torch.empty(0, dtype=int_type)
        self.winner_cells = torch.empty(0, dtype=int_type)
        self.predicted_cells = torch.empty(0, dtype=int_type)
        self.predicted_active_cells = torch.empty(0, dtype=int_type)
        self.active_basal_segments = torch.empty(0, dtype=int_type)
        self.active_apical_segments = torch.empty(0, dtype=int_type)
        self.matching_basal_segments = torch.empty(0, dtype=int_type)
        self.matching_apical_segments = torch.empty(0, dtype=int_type)
        self.basal_potential_overlaps = torch.empty(0, dtype=int_type)
        self.apical_potential_overlaps = torch.empty(0, dtype=int_type)

        self.use_apical_tiebreak = True
        self.use_apical_modulation_basal_threshold = True

        # one-to-one mapping of segment to cell
        self.apical_segment_to_cell = defaultdict(lambda : -1)
        self.basal_segment_to_cell = defaultdict(lambda : -1)

        # one-to-many mapping of cell to segments
        self.apical_cell_to_segments = defaultdict(lambda : torch.Tensor().to(int_type))
        self.basal_cell_to_segments = defaultdict(lambda : torch.Tensor().to(int_type))

    def reset(self):
        """
        clear all cell and segment activity.
        """

        self.active_cells = torch.empty(0, dtype=int_type)
        self.winner_cells = torch.empty(0, dtype=int_type)
        self.predicted_cells = torch.empty(0, dtype=int_type)
        self.predicted_active_cells = torch.empty(0, dtype=int_type)
        self.active_basal_segments = torch.empty(0, dtype=int_type)
        self.active_apical_segments = torch.empty(0, dtype=int_type)
        self.matching_basal_segments = torch.empty(0, dtype=int_type)
        self.matching_apical_segments = torch.empty(0, dtype=int_type)
        self.basal_potential_overlaps = torch.empty(0, dtype=int_type)
        self.apical_potential_overlaps = torch.empty(0, dtype=int_type)

    def depolarize_cells(self, basal_input, apical_input, learn):
        """
        calculate predictions.

        basal_input (torch.Tensor) contains the active input bits for the basal
        dendrite segments.

        apical_input (torch.Tensor) contains the active input bits for the apical
        dendrite segments.
        """

        # compute apical segment activity
        (active_apical_segments,
         matching_apical_segments,
         apical_potential_overlaps) = self.compute_apical_segment_activity(apical_input)

        if learn or not self.use_apical_modulation_basal_threshold:
            reduced_threshold_basal_cells = torch.Tensor()
        else:
            # map each segment to a cell
            reduced_threshold_basal_cells = self.map_apical_segments_to_cells(
                self.active_apical_segments
            )

        # compute basal segment activity
        (active_basal_segments,
         matching_basal_segments,
         basal_potential_overlaps) = self.compute_basal_segment_activity(
            basal_input,
            reduced_threshold_basal_cells
        )

        self.predicted_cells = self.compute_predicted_cells(active_basal_segments,
                                                              active_apical_segments)
        self.active_basal_segments = active_basal_segments
        self.active_apical_segments = active_apical_segments
        self.matching_basal_segments = matching_basal_segments
        self.matching_apical_segments = matching_apical_segments
        self.basal_potential_overlaps = basal_potential_overlaps
        self.apical_potential_overlaps = apical_potential_overlaps

    def activate_cells(
        self,
        active_minicolumns,
        basal_reinforce_candidates,
        apical_reinforce_candidates,
        basal_growth_candidates,
        apical_growth_candidates,
        learn=True
    ):
        """
        activate cells in specified minicolumns using result `depolarize_cells()` as
        predictions. then learn.

        active_minicolumns (torch.Tensor) is a list of active minicolumns.

        basal_reinforce_candidates (torch.Tensor) is a list of bits that the active
        cells may reinforce basal synapses to.

        apical_reinforce_candidates (torch.Tensor) is a list of bits that the active
        cells may reinforce apical synapses to.

        basal_growth_candidates (torch.Tensor) is a list of bits that the active cells
        may grow new basal synapses to.

        apical_growth_candidates (torch.Tensor) is a list of bits that the active cells
        may grow new apical synapses to.

        learn (bool) -- whether to grow, reinforce, and/or punish synapses
        """

        # correctly predicted cells are predicted and correspond to active_minicolumns
        correctly_predicted_cells = self.predicted_cellss[
            isin(
                self.predicted_cells // self.num_cells_per_minicolumn,
                active_minicolumns
            )
        ]

        # bursting minicolumns are all the active minicolumns that didn't have
        # any predicted cells
        bursting_minicolumns = active_minicolumns[
            ~isin(
                active_minicolumns,
                self.predicted_cells // self.num_cells_per_minicolumn
            )
        ]

        # mark all correctly predicted cells for activation
        # mark all cells in bursting minicolumns for activation
        new_active_cells = torch.cat([
            correctly_predicted_cells,
            get_cells_in_minicolumn(bursting_minicolumns, self.num_cells_per_minicolumn)
        ])

        # compute basal learning
        (learning_active_basal_segments,
         learning_matching_basal_segments,
         basal_segments_to_punish,
         new_basal_segment_cells,
         learning_cells) = self.compute_basal_learning(
             active_minicolumns, bursting_minicolumns, correctly_predicted_cells
        )










    def compute_apical_segment_activity(self, apical_input):
        """
        compute the active and matching apical segments for this timestep.

        apical_input (torch.Tensor) contains active input bits for the apical dendrite
        segments.

        returns:

        active_apical_segments (torch.Tensor) contains which dendrite segments have
        enough active connected synapses to cause a dendritic spike.

        matching_apical_segments (torch.Tensor) contains which dendrite segments have
        enough potential synapses to be selected for learning in a bursting minicolumn.

        apical_potential_overlaps (torch.Tensor) contains # of active potential synapses
        for each segment. includes counts for active, matching, and non-matching
        segments.
        """

        # compute active segments
        overlaps = (
            (self.apical_connections >= self.connected_permanence).to(real_type) \
                @ apical_input
        ).squeeze().to(int_type)

        active_apical_segments = torch.nonzero(
            overlaps >= self.activation_threshold
        ).squeeze().to(int_type)

        # compute matching segments
        apical_potential_overlaps = (
            (self.apical_connections > 0).to(real_type) @ apical_input
        ).squeeze().to(int_type)

        matching_apical_segments = torch.nonzero(
            apical_potential_overlaps >= self.matching_threshold
        ).squeeze().to(int_type)

        return (active_apical_segments, matching_apical_segments,
                apical_potential_overlaps)

    def compute_basal_segment_activity(self,
                                       basal_input,
                                       reduced_threshold_basal_cells):
        """
        compute the active and matching basal segments for this timestep.

        basal_input (torch.Tensor) contains active input bits for basal dendrite
        segments.

        reduced_threshold_basal_cells (torch.Tensor) contains cells with active apical
        segments -- these have a lower activation threshold for their basal segments
        (set by the reduced_basal_threshold parameter).

        returns:

        active_basal_segments (torch.Tensor) contains which dendrite segments have
        enough active connected synapses to cause a dendritic spike.

        matching_basal_segments (torch.Tensor) contains which dendrite segments have
        enough potential synapses to be selected for learning in a bursting minicolumn.

        basal_potential_overlaps (torch.Tensor) contains # of active potential synapses
        for each segment. includes counts for active, matching, and non-matching
        segments.
        """

        # compute active segments
        overlaps = (
            (self.basal_connections >= self.connected_permanence).to(real_type) \
                @ basal_input
        ).squeeze().to(int_type)

        # fully active basal segments (i.e. above the activation threshold)
        fully_active_basal_segments = torch.nonzero(
            overlaps >= self.activation_threshold
        ).squeeze().to(int_type)

        if (self.reduced_basal_threshold != self.activation_threshold and
            len(reduced_threshold_basal_cells) > 0):
            # active apical segments lower the activation threshold for
            # basal (lateral) segments. find segments that are above the reduced
            # threshold.
            potentially_active_basal_segments = torch.nonzero(
                (overlaps < self.activation_threshold) &
                (overlaps >= self.reduced_basal_threshold)
            )

            # find cells that correspond to each potentially active basal segment
            potentially_active_cells = self.map_basal_segments_to_cells(
                potentially_active_basal_segments
            )

            # pick cells that are potentially active and have reduced threshold basal
            # segments. then find the segments for these cells. add those segments
            # to the list of fully active basal segments.
            #
            # `isin()` method equivalent to `torch.isin(a, b)`, where
            # `a = potentially_active_cells` and `b = reduced_threshold_basal_cells`
            active_basal_segments = torch.cat([
                fully_active_basal_segments,
                potentially_active_basal_segments[
                    isin(potentially_active_cells, reduced_threshold_basal_cells)
                ]
            ])
        else:
            active_basal_segments = fully_active_basal_segments

        # compute matching segments
        basal_potential_overlaps = (
            (self.basal_connections > 0).to(real_type) @ basal_input
        ).squeeze().to(int_type)

        matching_basal_segments = torch.nonzero(
            basal_potential_overlaps >= self.matching_threshold
        ).squeeze().to(int_type)

        return (active_basal_segments, matching_basal_segments,
                basal_potential_overlaps)

    def compute_predicted_cells(self, active_basal_segments, active_apical_segments):
        """
        compute the predicted cells, given the set of active segments:

        an active basal segment is enough to predict a cell.
        an active apical segment is *not* enough to predict a cell.

        when a cell has both types of segments active, other cells in its minicolumn
        must also have both types of segments to be considered predictive.

        active_basal_segments (torch.Tensor) contains list of active basal segments.
        active_apical_segments (torch.Tensor) contains list of active apical segments.

        returns:

        predicted cells (torch.Tensor) contains list of predicted cells.
        """

        cells_for_basal_segments = self.map_basal_segments_to_cells(
            active_basal_segments
        )

        # if not using apical tiebreak, predicted cells = cells_for_basal_segments
        if not self.use_apical_tiebreak:
            return cells_for_basal_segments

        cells_for_apical_segments = self.map_apical_segments_to_cells(
            active_apical_segments
        )

        # fully depolarized cells should have both active basal and apical segments
        fully_depolarized_cells = intersection(cells_for_basal_segments,
                                               cells_for_apical_segments)

        # partly depolarized cells have active basal segments *but not* active apical
        # segments
        partly_depolarized_cells = difference(cells_for_basal_segments,
                                              fully_depolarized_cells)

        # choose which partly depolarized cells to inhibit
        inhibited_mask = isin(
            partly_depolarized_cells // self.num_cells_per_minicolumn,
            fully_depolarized_cells // self.num_cells_per_minicolumn
        )

        # predicted cells = fully_depolarized_cells + some of partly_depolarized_cells
        return torch.cat([
            fully_depolarized_cells,
            partly_depolarized_cells[~inhibited_mask]
        ])

    def compute_basal_learning(
        self,
        active_minicolumns,
        bursting_minicolumns,
        correctly_predicted_cells
    ):
        """
        correctly predicted cells always have active basal segments on which learning
        occurs.

        in bursting minicolumns, must learn on an existing basal segment or
        grow a new one.

        apical dendrites influence which cells are considered "predicted". therefore,
        an active apical dendrite can prevent some basal segments in active minicolumns
        from learning.

        returns:

        learning_active_basal_segments (torch.Tensor) contains active basal segments
        on correctly predicted cells.

        learning_matching_basal_segments (torch.Tensor) contains matching basal segments
        selected for learning in bursting minicolumns.

        basal_segments_to_punish (torch.Tensor) contains basal segments that should be
        punished for predicting an inactive mincolumn.

        learning_cells (torch.Tensor) contains cells that have learning basal segments
        or are selected to grow a basal segment.
        """

        # return subset of basal segments that are on the correctly predicted cells
        learning_active_basal_segments = self.active_basal_segments[
            isin(
                self.map_basal_segments_to_cells[self.active_basal_segments],
                correctly_predicted_cells
            )
        ]

        # find cells for matching basal segments
        cells_for_matching_basal_segments = self.map_basal_segments_to_cells(
            self.matching_basal_segments
        )

        # find unique cells which contain matching basal segments
        unique_cells_for_matching_basal_segments = torch.unique(
            cells_for_matching_basal_segments
        )

        # overlap between bursting minicolumns and minicolumns with cells
        # that have matching basal segments
        matching_cells_in_bursting_minicolumns \
        = unique_cells_for_matching_basal_segments[isin(
            unique_cells_for_matching_basal_segments // self.num_cells_per_minicolumn,
        )]

        # which bursting minicolumns contain no cells with matching basal segments
        bursting_minicolumns_with_no_matching_cells = bursting_minicolumns[~isin(
            bursting_minicolumns,
            unique_cells_for_matching_basal_segments // self.num_cells_per_minicolumn
        )]

        learning_matching_basal_segments = self.chooseBestSegmentPerColumn(
            self.basal_connections, matching_cells_in_bursting_minicolumns,
            self.matching_basal_segments, self.basal_potential_overlaps,
            self.num_cells_per_minicolumn
        )

        # for all the minicolumns covered by matching cells, choose each minicolumn's
        # matching segment with largest number of active potential synapses. pick
        # the first segment when there's a tie.
        candidate_matching_basal_segments = self.matching_basal_segments[
            isin(
                self.map_basal_segments_to_cells[self.matching_basal_segments],
                matching_cells_in_bursting_minicolumns
            )
        ]

        # 
        cell_scores = self.basal_potential_overlaps[candidate_matching_basal_segments]


        candidate_minicolumns_with_matching_basal_segments \
        = self.map_basal_segments_to_cells[
            candidate_matching_basal_segments
        ] // self.num_cells_per_minicolumn

        # 
         =  argmax_multi(
                # 
                self.basal_potential_overlaps[candidate_matching_basal_segments],

                # candidate minicolumns with matching basal segments
                self.map_basal_segments_to_cells[
                    candidate_matching_basal_segments
                ] // self.num_cells_per_minicolumn
        )

        

        















    def map_apical_segments_to_cells(self, segments):
        """
        map each apical segment in `segments` (torch.Tensor) to a cell.

        mapping is one-to-one.
        """

        return segments.clone().detach().apply_(
            lambda segment : self.apical_cell_to_segments[segment]
        ).to(int_type)

    def map_basal_segments_to_cells(self, segments):
        """
        map each basal segment in `segments` (torch.Tensor) to a cell.

        mapping is one-to-one.
        """

        return segments.clone().detach().apply_(
            lambda segment : self.basal_segment_to_cell[segment]
        ).to(int_type)


def isin(a, b):
    """
    return True for each element of `a` (torch.Tensor) if it is
    in `b` (torch.Tensor).

    both `a` and `b` are 1D tensors.
    """

    return (a.view(-1, 1) == b).any(axis=-1)

def intersection(a, b):
    """
    returns intersection of elements in `a` (torch.Tensor) and `b` (torch.Tensor).

    both `a` and `b` are 1D tensors.
    both `a` and `b` cannot have any duplicate elements.
    """

    uniques, counts = torch.cat([a, b]).unique(return_counts=True)

    return uniques[counts > 1]

def difference(a, b):
    """
    returns unique values in `a` (torch.Tensor) that are not in `b` (torch.Tensor).

    both `a` and `b` are 1D tensors.
    both `a` and `b` cannot have any duplicate elements.
    """

    expanded_b = b.expand(a.shape[0], b.shape[0]).T

    return a[(a != expanded_b).T.prod(axis=1) == 1]

def get_cells_in_minicolumn(minicolumns, num_cells_per_minicolumn):
    """
    calculate all cell indices in the specified minicolumns.

    minicolumns (torch.Tensor) contains all minicolumns.
    cells_per_minicolumn (int) is number of cells per minicolumn.
    """

    return (minicolumns * num_cells_per_minicolumn).view(-1, 1) + \
        torch.arange(num_cells_per_minicolumn).to(int_type).flatten()

def argmax_multi(tensor, groups):
    """
    gets indices of the max values of each group in `tensor` (torch.Tensor), 
    grouping the elements by their correspondiing value in `groups` (torch.Tensor). 

    returns index (within `tensor`) of maximal element in each group.

    example: 
        argmax_multi(tensor = [5, 4, 7, 2, 9, 8],    -->    [2, 4]   
                     groups = [0, 0, 0, 1, 1, 1])
    """

    # find the set of all groups 
    values = torch.unique(groups)
    
    # non-zero elements in column `i` of `max_values` contain values of 
    # `tensor` that belong in group `groups[i]`.
    #
    # (add a `1` in order to represent zeros.)
    max_values = (tensor + 1).view(-1, 1) * (groups.view(-1, 1) == values)

    # return indices of maximal element in each group. 
    # break ties by picking the first occurrence of each maximal value.
    return torch.max(max_values, dim=0).indices



