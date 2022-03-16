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
from functools import reduce

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
        basal_segment_incorrect_decrement=0.0,
        apical_segment_incorrect_decrement=0.0,
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

        basal_segment_incorrect_decrement:          amount by which permanences are
                                                    punished for incorrect predictions.
                                                    default `0.0`.

        apical_segment_incorrect_decrement:         amount by which segments are
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
        self.basal_segment_incorrect_decrement = basal_segment_incorrect_decrement
        self.apical_segment_incorrect_decrement = apical_segment_incorrect_decrement
        self.max_synapses_per_segment = max_synapses_per_segment

        # random seed
        if seed == -1:
            self.seed = torch.random.seed()
        else:
            self.seed = seed

        self.generator = torch.random.manual_seed(self.seed)

        self.basal_connections = torch.zeros(
            (0, self.basal_input_size),
            dtype=real_type
        )

        self.apical_connections = torch.zeros(
            (0, self.apical_input_size), 
            dtype=real_type
        )

        self.num_total_cells = self.num_minicolumns * self.num_cells_per_minicolumn 


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
        self.cell_to_apical_segments = defaultdict(set)
        self.cell_to_basal_segments = defaultdict(set)

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

        self.predicted_cells = self.compute_predicted_cells(
            active_basal_segments,
            active_apical_segments
        )

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
        correctly_predicted_cells = self.predicted_cells[
            isin(
                cells_to_minicolumns(
                    self.predicted_cells, self.num_cells_per_minicolumn
                ),
                active_minicolumns
            )
        ]

        # bursting minicolumns are all the active minicolumns that didn't have
        # any predicted cells
        bursting_minicolumns = active_minicolumns[
            ~isin(
                active_minicolumns,
                cells_to_minicolumns(
                    self.predicted_cells, self.num_cells_per_minicolumn
                )
            )
        ]

        # newly_active_cells:
        #   - all correctly predicted cells
        #   - all cells in bursting minicolumns
        newly_active_cells = torch.cat([
            correctly_predicted_cells,
            get_cells_in_minicolumns(
                bursting_minicolumns, self.num_cells_per_minicolumn
            )
        ])

        # compute basal learning
        (learning_active_basal_segments,
         learning_matching_basal_segments,
         basal_segments_to_punish,
         cells_with_new_basal_segments,
         all_learning_cells) = self.compute_basal_learning(
             active_minicolumns, bursting_minicolumns, correctly_predicted_cells
        )

        # compute apical learning
        (learning_active_apical_segments,
         learning_matching_apical_segments,
         apical_segments_to_punish, 
         cells_with_new_apical_segments) = self.compute_apical_learning(
             all_learning_cells, active_minicolumns
        )

        if learn:
            # learning process for basal segments
            for learning_segments in (learning_active_basal_segments,
                                      learning_matching_basal_segments):
                self.learn_basal_synapses(
                    learning_segments, 
                    basal_reinforce_candidates, 
                    basal_growth_candidates
                )

            # learning process for apical segments
            for learning_segments in (learning_active_apical_segments, 
                                      learning_matching_apical_segments):
                self.learn_apical_synapses(
                    learning_segments, 
                    apical_reinforce_candidates,
                    apical_growth_candidates
                )

            if self.basal_segment_incorrect_decrement != 0.0:
                self.adjust_synapses_on_basal_segments(
                    basal_segments_to_punish,
                    basal_reinforce_candidates,
                    -self.basal_segment_incorrect_decrement
                )
            
            if self.apical_segment_incorrect_decrement != 0.0:
                self.adjust_synapses_on_apical_segments(
                    apical_segments_to_punish,
                    apical_reinforce_candidates,
                    -self.apical_segment_incorrect_decrement
                )
            
            # grow new basal segments
            if basal_growth_candidates.numel() > 0:
                self.learn_basal_segments(
                    cells_with_new_basal_segments, 
                    basal_growth_candidates
                )
            
            # grow new apical segments
            if apical_growth_candidates.numel() > 0:
                self.learn_apical_segments(
                    cells_with_new_apical_segments,
                    apical_growth_candidates
                )
        
        self.active_cells = torch.sort(newly_active_cells)
        self.winner_cells = torch.sort(all_learning_cells)
        self.predicted_active_cells = correctly_predicted_cells


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

        apical_input_sdr = torch.zeros(self.apical_input_size)
        apical_input_sdr[apical_input.to(int_type)] = 1

        # compute active segments
        overlaps = (
            (self.apical_connections >= self.connected_permanence).to(real_type) \
                @ apical_input_sdr
        ).squeeze().to(int_type)

        active_apical_segments = torch.nonzero(
            overlaps >= self.activation_threshold
        ).squeeze().to(int_type)

        # compute matching segments
        apical_potential_overlaps = (
            (self.apical_connections > 0).to(real_type) @ apical_input_sdr
        ).squeeze().to(int_type)

        matching_apical_segments = torch.nonzero(
            apical_potential_overlaps >= self.matching_threshold
        ).squeeze().to(int_type)

        return (active_apical_segments, matching_apical_segments,
                apical_potential_overlaps)

    def compute_basal_segment_activity(
        self,
        basal_input,
        reduced_threshold_basal_cells
    ):
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

        basal_input_sdr = torch.zeros(self.basal_input_size)
        basal_input_sdr[basal_input.to(int_type)] = 1

        # compute active segments
        overlaps = (
            (self.basal_connections >= self.connected_permanence).to(real_type) \
                @ basal_input_sdr
        ).squeeze().to(int_type)

        # fully active basal segments (i.e. above the activation threshold)
        fully_active_basal_segments = torch.nonzero(
            overlaps >= self.activation_threshold
        ).squeeze().to(int_type)

        if (self.reduced_basal_threshold != self.activation_threshold and
            reduced_threshold_basal_cells.numel() > 0):
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

        basal_potential_overlaps = (
            (self.basal_connections > 0).to(real_type) @ basal_input_sdr
        ).squeeze().to(int_type)

        # compute matching segments
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

        cells_with_basal_segments = self.map_basal_segments_to_cells(
            active_basal_segments
        )

        # if not using apical tiebreak, predicted cells = cells_with_basal_segments
        if not self.use_apical_tiebreak:
            return cells_with_basal_segments

        cells_with_apical_segments = self.map_apical_segments_to_cells(
            active_apical_segments
        )

        # fully depolarized cells should have both active basal and apical segments
        fully_depolarized_cells = intersection(cells_with_basal_segments,
                                               cells_with_apical_segments)

        # partly depolarized cells have active basal segments *but not* active apical
        # segments
        partly_depolarized_cells = difference(cells_with_basal_segments,
                                              fully_depolarized_cells)

        # choose which partly depolarized cells to inhibit
        inhibited_mask = isin(
            cells_to_minicolumns(
                partly_depolarized_cells, self.num_cells_per_minicolumn
            ),
            cells_to_minicolumns(
                fully_depolarized_cells, self.num_cells_per_minicolumn
            )
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

        cells_with_new_basal_segments (torch.Tensor) contains cells in bursting 
        minicolumns that are selected to grow new basal segments.

        all_learning_cells (torch.Tensor) contains cells that have learning basal 
        segments or are selected to grow a basal segment.
        """

        # ***** LEARNING_ACTIVE_BASAL_SEGMENTS ***** #

        # return subset of basal segments that are on the correctly predicted cells
        learning_active_basal_segments = self.active_basal_segments[
            isin(
                self.map_basal_segments_to_cells(self.active_basal_segments),
                correctly_predicted_cells
            )
        ]

        # ***** BASAL_SEGMENTS_TO_PUNISH ***** #

        # find cells with matching basal segments
        cells_with_matching_basal_segments = self.map_basal_segments_to_cells(
            self.matching_basal_segments
        )

        # basal segments to punish are not associated with cells in any 
        # active minicolumns
        # (i.e. these apical segments incorrectly predicted minicolumns)
        basal_segments_to_punish = self.matching_basal_segments[
            ~isin(
                cells_to_minicolumns(
                    cells_with_matching_basal_segments, self.num_cells_per_minicolumn
                ),
                active_minicolumns
            )
        ]

        # ***** LEARNING_MATCHING_BASAL_SEGMENTS ***** #

        # find unique cells which contain matching basal segments
        unique_cells_with_matching_basal_segments = torch.unique(
            cells_with_matching_basal_segments
        )

        # overlap between (1) minicolumns with cells with matching basal segments 
        # and (2) bursting minicolumns 
        # ====> overlap = bursting minicolumns with cells with matching basal segments
        #
        # then get all the cells in the overlap
        cells_with_matching_basal_segments_in_bursting_minicolumns \
            = unique_cells_with_matching_basal_segments[
                isin(
                    cells_to_minicolumns(
                        unique_cells_with_matching_basal_segments, 
                        self.num_cells_per_minicolumn
                    ),
                    bursting_minicolumns
                )
            ]

        # find matching basal segments whose cells are in bursting minicolumns
        matching_basal_segments_of_cells_in_bursting_minicolumns = \
            self.matching_basal_segments[
                isin(
                    cells_with_matching_basal_segments,
                    cells_with_matching_basal_segments_in_bursting_minicolumns
                )
            ]

        # best matching basal segment (highest # of active synapses) 
        # among all cells in each bursting minicolumn
        learning_matching_basal_segments = \
            matching_basal_segments_of_cells_in_bursting_minicolumns[
                # for each minicolumn with cells with matching basal segments, 
                # choose ONE matching segment with the largest number of 
                # active potential synapses.
                # 
                # pick the first segment when there's a tie.
                argmax_multi(
                    # number of active potential synapses for matching basal segments 
                    # of cells in bursting minicolumns
                    values=self.basal_potential_overlaps[
                        matching_basal_segments_of_cells_in_bursting_minicolumns
                    ],

                    # which bursting minicolumn each cell with matching basal segments
                    # belongs to
                    groups=cells_to_minicolumns(
                        self.map_basal_segments_to_cells(
                            matching_basal_segments_of_cells_in_bursting_minicolumns
                        ),
                        self.num_cells_per_minicolumn
                    )
                )
            ]

        # ***** CELLS_WITH_NEW_BASAL_SEGMENTS ***** # 

        # which bursting minicolumns contain no cells with matching basal segments
        bursting_minicolumns_with_cells_with_no_matching_basal_segments = \
            bursting_minicolumns[
                ~isin(
                    bursting_minicolumns,
                    cells_to_minicolumns(
                        unique_cells_with_matching_basal_segments,
                        self.num_cells_per_minicolumn
                    )
                )
            ]

        # cells in bursting minicolumns with no matching basal segments
        #
        # arranged as minicolumns x cells.
        cells_in_bursting_minicolumns_with_no_matching_basal_segments = \
            get_cells_in_minicolumns(
                bursting_minicolumns_with_cells_with_no_matching_basal_segments, 
                self.num_cells_per_minicolumn
            )

        # segment counts for each cell in bursting minicolumns with no matching basal
        # segments. 
        #
        # arranged as minicolumns x cells. 
        basal_segment_counts = self.get_basal_segment_counts(
            cells_in_bursting_minicolumns_with_no_matching_basal_segments
        ).reshape(
            bursting_minicolumns_with_cells_with_no_matching_basal_segments.numel(),
            self.num_cells_per_minicolumn
        )

        min_basal_segment_counts = basal_segment_counts.amin(dim=1, keepdim=True)

        # reduce cells to those with fewest basal segments in each minicolumn
        cells_in_bursting_minicolumns_with_no_matching_basal_segments = \
            cells_in_bursting_minicolumns_with_no_matching_basal_segments[
                torch.nonzero(
                    (basal_segment_counts == min_basal_segment_counts).ravel()
                ).squeeze()
            ]

        # get unique bursting minicolumns with cells with no matching basal segments
        unique_minicolumns, num_candidates_in_minicolumns = torch.unique(
            cells_to_minicolumns(
                cells_in_bursting_minicolumns_with_no_matching_basal_segments,
                self.num_cells_per_minicolumn
            ), 
            return_counts=True
        )

        # only pick one cell per bursting minicolumn that has no matching basal segments
        one_cell_per_minicolumn_filter = (
                unique_minicolumns.view(-1, 1) \
                == cells_to_minicolumns(
                    cells_in_bursting_minicolumns_with_no_matching_basal_segments,
                    self.num_cells_per_minicolumn
                )
        ).to(int_type).argmax(dim=1)

        # randomly keep only some of the cells
        one_cell_per_minicolumn_filter = (
            one_cell_per_minicolumn_filter + torch.rand(
                size=(
                    one_cell_per_minicolumn_filter.numel(),
                ), 
                generator=self.generator
            ).to(real_type) * num_candidates_in_minicolumns
        ).floor()

        # cells with new basal segments 
        cells_with_new_basal_segments = \
            cells_in_bursting_minicolumns_with_no_matching_basal_segments[
                one_cell_per_minicolumn_filter
            ]

        # ***** ALL_LEARNING_CELLS ***** # 

        # all learning cells include:
        #   - correctly predicted cells
        #   - cells with learning with matching basal segments
        #   - cells with new basal segments
        all_learning_cells = torch.cat([
            correctly_predicted_cells,
            self.map_basal_segments_to_cells(learning_matching_basal_segments),
            cells_with_new_basal_segments
        ])

        return (learning_active_basal_segments, 
                learning_matching_basal_segments,
                basal_segments_to_punish,
                cells_with_new_basal_segments,
                all_learning_cells)

    def compute_apical_learning(self, all_learning_cells, active_minicolumns):
        """
        set of `all_learning_cells` (torch.Tensor) is determined from 
        basal segments. compute apical learning on the same cells. 

        learn on any active segments on learning cells. 
        for cells without active segments, learn on the best matching segment.
        for cells without a matching segment, grow a new segment.

        returns:

        learning_active_apical_segments (torch.Tensor) contains active apical segments
        on correctly predicted cells.

        learning_matching_apical_segments (torch.Tensor) contains matching apical
        segments selected for learning in bursting minicolumns.

        apical_segments_to_punish (torch.Tensor) contains apical segments that should
        be punished for predicting an inactive minicolumn.

        cells_with_new_apical_segments (torch.Tensor) contains cells in bursting 
        minicolumns that are selected to grow new apical segments.
        """

        # ***** LEARNING_ACTIVE_APICAL_SEGMENTS ***** #

        # return subset of apical segments that are on the learning cells
        learning_active_apical_segments = self.active_apical_segments[isin(
            self.map_apical_segments_to_cells(self.active_apical_segments),
            all_learning_cells
        )]

        # ***** LEARNING_MATCHING_APICAL_SEGMENTS ***** #
        
        # learning cells without active apical segments
        learning_cells_without_active_apical_segments = difference(
            all_learning_cells, 
            self.map_apical_segments_to_cells(learning_active_apical_segments)
        )

        cells_with_matching_apical_segments = self.map_apical_segments_to_cells(
            self.matching_apical_segments
        )

        # learning cells with matching apical segments are cells that have both 
        # active apical segments and matching apical segments
        learning_cells_with_matching_apical_segments = intersection(
            learning_cells_without_active_apical_segments,
            cells_with_matching_apical_segments
        )

        # find matching apical segments whose cells are learning
        matching_apical_segments_of_learning_cells = self.matching_apical_segments[
            isin(
                self.map_apical_segments_to_cells(self.matching_apical_segments),
                learning_cells_with_matching_apical_segments
            )
        ]

        # choose matching segment with largest number of active potential synapses
        learning_matching_apical_segments = matching_apical_segments_of_learning_cells[
            argmax_multi(
                values=self.apical_potential_overlaps[
                    matching_apical_segments_of_learning_cells
                ],
                groups=self.map_apical_segments_to_cells(
                    matching_apical_segments_of_learning_cells
                )
            )
        ]

        # ***** CELLS_WITH_NEW_APICAL_SEGMENTS ***** #

        # cells that need to grow a new apical segment
        cells_with_new_apical_segments = difference(
            learning_cells_without_active_apical_segments,
            learning_cells_with_matching_apical_segments
        )

        # ***** CELLS_WITH_NEW_APICAL_SEGMENTS ***** #

        # apical segments to punish are not associated with cells in any 
        # active minicolumns 
        # (i.e. these apical segments incorrectly predicted minicolumns)
        apical_segments_to_punish = self.matching_apical_segments[
            ~isin(
                cells_to_minicolumns(
                    cells_with_matching_apical_segments, self.num_cells_per_minicolumn
                ),
                active_minicolumns
            )
        ]

        return (learning_active_apical_segments,
                learning_matching_apical_segments,
                apical_segments_to_punish,
                cells_with_new_apical_segments)

    def learn_basal_synapses(
        self, 
        learning_segments, 
        basal_reinforce_candidates,
        basal_growth_candidates
    ):
        """
        adjust synapse permanences, grow new synapses on basal segments.
        """

        # ***** ADJUST SYNAPSES ***** #

        # increment synapses 
        self.adjust_synapses_on_basal_segments(
            learning_segments, 
            basal_reinforce_candidates,
            self.permanence_increment
        )

        # decrement synapses
        self.adjust_synapses_on_basal_segments(
            learning_segments,
            difference(
                torch.arange(self.basal_connections.shape[1]),
                basal_reinforce_candidates
            ),
            -self.permanence_decrement
        )

        # ***** GROW NEW SYNAPSES ***** #

        if self.sample_size == -1:
            max_new_synapses = basal_growth_candidates.numel()
        else:
            # sample size (how much of active SDR to sample with synapses) - 
            # how much of each basal learning segment is currently sampled with synapses
            max_new_synapses = self.sample_size - \
                               self.basal_potential_overlaps[learning_segments]
        
        # if defining the maximum number of synapses per segment
        if self.max_synapses_per_segment != -1:
            # how many active synapses per cell
            synapse_counts = self.basal_connections.count_nonzero(dim=1)

            # max synapses left to grow is max synapses allowed per segment - 
            # current active synapse counts (per cell) 
            max_synapses_to_reach = self.max_synapses_per_segment - synapse_counts
            
            # pick the smaller number of maximum new synapses to grow
            max_new_synapses = torch.where(max_new_synapses <= max_synapses_to_reach,
                                           max_new_synapses, max_synapses_to_reach)


        self.grow_synapses_on_basal_segments(
            learning_segments, basal_growth_candidates, max_new_synapses
        )
    
    def learn_apical_synapses(
        self, 
        learning_segments, 
        apical_reinforce_candidates,
        apical_growth_candidates
    ):
        """
        adjust synapse permanences, grow new synapses on apical segments.
        """

        # ***** ADJUST SYNAPSES ***** #

        # increment synapses 
        self.adjust_synapses_on_apical_segments(
            learning_segments, 
            apical_reinforce_candidates,
            self.permanence_increment
        )

        # decrement synapses
        self.adjust_synapses_on_apical_segments(
            learning_segments,
            difference(
                torch.arange(self.apical_connections.shape[1]),
                apical_reinforce_candidates
            ),
            -self.permanence_decrement
        )

        # ***** GROW NEW SYNAPSES ***** #

        if self.sample_size == -1:
            max_new_synapses = apical_growth_candidates.numel()
        else:
            # sample size (how much of active SDR to sample with synapses) - 
            # how much of each apical learning segment is currently sampled with 
            # synapses
            max_new_synapses = self.sample_size - \
                               self.apical_potential_overlaps[learning_segments]
        
        # if defining the maximum number of synapses per segment
        if self.max_synapses_per_segment != -1:
            # how many active synapses per cell
            synapse_counts = self.apical_connections.count_nonzero(dim=1)

            # max synapses left to grow is max synapses allowed per segment - 
            # current active synapse counts (per cell) 
            max_synapses_to_reach = self.max_synapses_per_segment - synapse_counts
            
            # pick the smaller number of maximum new synapses to grow
            max_new_synapses = torch.where(max_new_synapses <= max_synapses_to_reach,
                                           max_new_synapses, max_synapses_to_reach)

        self.grow_synapses_on_apical_segments(
            learning_segments, apical_growth_candidates, max_new_synapses
        )

    def grow_synapses_on_basal_segments(
        self, 
        segments, 
        active_inputs, 
        max_new_synapses
    ):
        """
        grow synapses on `segments` (torch.Tensor), which lists the 
        relevant basal segments. 

        `active_inputs` (torch.Tensor) specifies list of bits that the active
        cells may reinforce basal synapses to.

        `max_new_synapses` (int or torch.Tensor) specifies maximum number of new 
        synapses to be created per row.
        """

        if isinstance(max_new_synapses, int):
            max_new_synapses = torch.Tensor([max_new_synapses]).repeat(segments.numel())

        inds = torch.cartesian_prod(segments, active_inputs).long()
        ind_x = inds[:, 0]
        ind_y = inds[:, 1]

        zero_ind = (self.basal_connections[ind_x, ind_y] == 0).nonzero()

        # number of zeros per row
        split_ind = (
            ~torch.stack(
                self.basal_connections[ind_x, ind_y].chunk(segments.numel())
            ).to(torch.bool)
        ).sum(dim=1).cumsum(dim=0) - 1

        # only consider zero elements
        ind_x = ind_x[zero_ind].squeeze()
        ind_y = ind_y[zero_ind].squeeze()

        mask = torch.cat(
            ((ind_x == ind_x.view(-1, 1))[split_ind], max_new_synapses.view(-1, 1)),
            dim=1
        )

        # indices (randomly chosen) at which to initialize new synapses
        change_inds = torch.cat(list(
            map(lambda x : torch.multinomial(
                    x.squeeze()[:-1], 
                    num_samples=int(x.squeeze()[-1].item()),
                    generator=self.generator),
                mask.chunk(mask.shape[0])
            )
        ))

        # initialize synapses
        self.basal_connections[
            ind_x[change_inds], ind_y[change_inds]
        ] = self.initial_permanence

        self.basal_connections[
            ind_x[change_inds]
        ] = self.basal_connections[ind_x[change_inds]].clamp(0, 1)
        

    def grow_synapses_on_apical_segments(
        self,
        segments,
        active_inputs,
        max_new_synapses
    ):
        """
        grow synapses on `segments` (torch.Tensor), which lists the 
        relevant apical segments. 

        `active_inputs` (torch.Tensor) specifies list of bits that the active
        cells may reinforce apical synapses to.

        `max_new_synapses` (torch.Tensor) specifies maximum number of new synapses to
        be created per row.
        """
        
        if isinstance(max_new_synapses, int):
            max_new_synapses = torch.Tensor([max_new_synapses]).repeat(segments.numel())

        inds = torch.cartesian_prod(segments, active_inputs).long()
        ind_x = inds[:, 0]
        ind_y = inds[:, 1]

        zero_ind = (self.apical_connections[ind_x, ind_y] == 0).nonzero()

        # number of zeros per row
        split_ind = (
            ~torch.stack(
                self.apical_connections[ind_x, ind_y].chunk(segments.numel())
            ).to(torch.bool)
        ).sum(dim=1).cumsum(dim=0) - 1

        # only consider zero elements
        ind_x = ind_x[zero_ind].squeeze()
        ind_y = ind_y[zero_ind].squeeze()

        mask = torch.cat(
            ((ind_x == ind_x.view(-1, 1))[split_ind], max_new_synapses.view(-1, 1)),
            dim=1
        )

        # indices (randomly chosen) at which to initialize new synapses
        change_inds = torch.cat(list(
            map(lambda x : torch.multinomial(
                    x.squeeze()[:-1], 
                    num_samples=int(x.squeeze()[-1].item()),
                    generator=self.generator),
                mask.chunk(mask.shape[0])
            )
        ))

        # initialize synapses
        self.apical_connections[
            ind_x[change_inds], ind_y[change_inds]
        ] = self.initial_permanence

        self.apical_connections[
            ind_x[change_inds]
        ] = self.apical_connections[ind_x[change_inds]].clamp(0, 1)

    def learn_basal_segments(
        self, 
        cells_with_new_basal_segments, 
        basal_growth_candidates
    ):
        """
        grow new basal segments on `cells_with_new_basal_segments` (torch.Tensor), 
        given `basal_growth_candidates` (torch.Tensor), which is a list of bits 
        that the active cells may grow new basal synapses to.
        """
        
        num_new_synapses = basal_growth_candidates.numel()

        if self.sample_size != -1:
            num_new_synapses = min(num_new_synapses, self.sample_size)
        
        if self.max_synapses_per_segment != -1:
            num_new_synapses = min(num_new_synapses, self.max_synapses_per_segment)
        
        # new basal segment id's
        new_basal_segments = range(
            self.basal_connections.shape[0], 
            self.basal_connections.shape[0] + cells_with_new_basal_segments.numel()
        )

        # update cell <--> basal segment mappings
        for cell, segment in zip(
            cells_with_new_basal_segments.tolist(), new_basal_segments
        ):
            self.basal_segment_to_cell[segment] = cell
            self.cell_to_basal_segments[cell].add(segment)
        
        # create new basal segments in permanence matrix
        self.basal_connections = torch.cat(
            (
                self.basal_connections, 
                torch.zeros(
                    (len(new_basal_segments), self.basal_input_size), 
                    dtype=int_type
                )
            )
        )

        # grow synapses on new basal segments
        self.grow_synapses_on_basal_segments(
            new_basal_segments, basal_growth_candidates, num_new_synapses
        )

    def learn_apical_segments(
        self,
        cells_with_new_apical_segments,
        apical_growth_candidates
    ):
        """
        grow new apical segments on `cells_with_new_apical_segments` (torch.Tensor), 
        given `apical_growth_candidates` (torch.Tensor), which is a list of bits 
        that the active cells may grow new apical synapses to.
        """

        num_new_synapses = apical_growth_candidates.numel()

        if self.sample_size != -1:
            num_new_synapses = min(num_new_synapses, self.sample_size)
        
        if self.max_synapses_per_segment != -1:
            num_new_synapses = min(num_new_synapses, self.max_synapses_per_segment)
        
        # new apical segment id's
        new_apical_segments = range(
            self.apical_connections.shape[0], 
            self.apical_connections.shape[0] + cells_with_new_apical_segments.numel()
        )

        # update cell <--> apical segment mappings
        for cell, segment in zip(
            cells_with_new_apical_segments.tolist(), new_apical_segments
        ):
            self.apical_segment_to_cell[segment] = cell
            self.cell_to_apical_segments[cell].add(segment)
        
        # create new apical segments in permanence matrix
        self.apical_connections = torch.cat(
            (
                self.apical_connections, 
                torch.zeros(
                    (len(new_apical_segments), self.apical_input_size), 
                    dtype=int_type
                )
            )
        )

        # grow synapses on new apical segments
        self.grow_synapses_on_apical_segments(
            new_apical_segments, apical_growth_candidates, num_new_synapses
        )

    def adjust_synapses_on_basal_segments(self, segments, active_inputs, delta):
        """
        adjust synapses on basal segments. 

        `segments` (torch.Tensor) specifies which segments to consider. 

        `active_inputs` (torch.Tensor) specifies list of bits that the active
        cells may reinforce basal synapses to.

        `delta` (int) specifies how much each synapse is strengthened/weakened.
        """

        ind = torch.cartesian_prod(segments, active_inputs).to(int_type)
        ind_x = ind[:, 0]
        ind_y = ind[:, 1]
        nz_ind = self.basal_connections[ind_x, ind_y].nonzero()

        self.basal_connections[ind_x[nz_ind], ind_y[nz_ind]] += delta

        self.basal_connections[ind_x[nz_ind]] = \
            self.basal_connections[ind_x[nz_ind]].clamp(0, 1)

    def adjust_synapses_on_apical_segments(self, segments, active_inputs, delta):
        """
        adjust synapses on apical segments. 

        `segments` (torch.Tensor) specifies which segments to consider. 

        `active_inputs` (torch.Tensor) specifies list of bits that the active
        cells may reinforce apical synapses to.

        `delta` (int) specifies how much each synapse is strengthened/weakened.
        """

        ind = torch.cartesian_prod(segments, active_inputs).to(int_type)
        ind_x = ind[:, 0]
        ind_y = ind[:, 1]
        nz_ind = self.apical_connections[ind_x, ind_y].nonzero()

        self.apical_connections[ind_x[nz_ind], ind_y[nz_ind]] += delta

        self.apical_connections[ind_x[nz_ind]] = \
            self.apical_connections[ind_x[nz_ind]].clamp(0, 1)

    def map_apical_segments_to_cells(self, segments):
        """
        map each apical segment in `segments` (torch.Tensor) to a cell.

        mapping is one-to-one.
        """

        return segments.clone().detach().apply_(
            lambda segment : self.apical_segment_to_cell[segment]
        ).to(int_type)

    def map_basal_segments_to_cells(self, segments):
        """
        map each basal segment in `segments` (torch.Tensor) to a cell.

        mapping is one-to-one.
        """

        return segments.clone().detach().apply_(
            lambda segment : self.basal_segment_to_cell[segment]
        ).to(int_type)

    def get_basal_segment_counts(self, cells):
        """
        return number of basal segments for each cell in `cells` (torch.Tensor)
        """

        return cells.clone().detach().apply_(
            lambda cell : len(self.cell_to_basal_segments[cell])
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

def get_cells_in_minicolumns(minicolumns, num_cells_per_minicolumn):
    """
    calculate all cell indices in the specified minicolumns.

    minicolumns (torch.Tensor) contains all minicolumns.
    cells_per_minicolumn (int) is number of cells per minicolumn.
    """

    return ((minicolumns * num_cells_per_minicolumn).view(-1, 1) + \
        torch.arange(num_cells_per_minicolumn).to(int_type)).flatten()

def cells_to_minicolumns(cells, num_cells_per_minicolumn):
    """
    return minicolumn indices (torch.Tensor) for `cells` (torch.Tensor). 
    """
    
    return cells.div(num_cells_per_minicolumn, rounding_mode="floor")

def argmax_multi(values, groups):
    """
    gets indices of the max values of each group in `values` (torch.Tensor), 
    grouping the elements by their correspondiing value in `groups` (torch.Tensor). 

    returns index (within `values`) of maximal element in each group.

    example: 
        argmax_multi(values = [5, 4, 7, 2, 9, 8],    -->    [2, 4]   
                     groups = [0, 0, 0, 1, 1, 1])
    """

    # find the set of all groups 
    unique_groups = torch.unique(groups)
    
    # non-zero elements in column `i` of `max_values` contain elements of 
    # `values` that belong in group `groups[i]`.
    #
    # (add a `1` in order to represent zeros.)
    max_values = (values + 1).view(-1, 1) * (groups.view(-1, 1) == unique_groups)

    # return indices of maximal element in each group. 
    # break ties by picking the first occurrence of each maximal value.
    return torch.max(max_values, dim=0).indices


class SequenceMemoryApicalTiebreak(TemporalMemoryApicalTiebreak):
    """
    sequence memory with apical tiebreak.
    """

    def __init__(
        self,
        num_minicolumns=2048,
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
        basal_segment_incorrect_decrement=0.0,
        apical_segment_incorrect_decrement=0.0,
        max_synapses_per_segment=-1,
        seed=42
    ):

        params = {
            "num_minicolumns" : num_minicolumns,
            "basal_input_size" : num_minicolumns * num_cells_per_minicolumn,
            "apical_input_size" : apical_input_size,
            "num_cells_per_minicolumn" : num_cells_per_minicolumn,
            "activation_threshold" : activation_threshold,
            "reduced_basal_threshold" : reduced_basal_threshold,
            "initial_permanence" : initial_permanence,
            "connected_permanence" : connected_permanence,
            "matching_threshold" : matching_threshold,
            "sample_size" : sample_size,
            "permanence_increment" : permanence_increment,
            "permanence_decrement" : permanence_decrement,
            "basal_segment_incorrect_decrement" : basal_segment_incorrect_decrement,
            "apical_segment_incorrect_decrement" : apical_segment_incorrect_decrement,
            "max_synapses_per_segment" : max_synapses_per_segment,
            "seed" : seed
        }

        super().__init__(**params)

        self.previous_apical_input = torch.empty(0, dtype=int_type)
        self.previous_apical_growth_candidates = torch.empty(0, dtype=int_type)
        self.previous_predicted_cells = torch.empty(0, dtype=int_type)
    
    def reset(self):
        """
        clear all cell and segment activity.
        """

        super().reset()

        self.previous_apical_input = torch.empty(0, dtype=int_type)
        self.previous_apical_growth_candidates = torch.empty(0, dtype=int_type)
        self.previous_predicted_cells = torch.empty(0, dtype=int_type)

    def compute(
        self,
        active_minicolumns,
        apical_input=(),
        apical_growth_candidates=None,
        learn=True
    ):
        """
        perform one timestep: activate the specified minicolumn using the predictions
        from the previous timestep and then learn. form a new set of predictions 
        using the newly active cells and the apical input. 

        `active_minicolumns` (torch.Tensor) contains the active minicolumns.

        `apical_input` (torch.Tensor) contains the list of active input bits
        for the apical dendrite segments.

        `apical_growth_candidates` (torch.Tensor or None) contains the list of bits 
        that the active cells may grow new apical synapses to. 
        if None, the `apical_input` is assumed to be growth candidates.

        `learn` (bool) -- whether to grow / reinforce / punish synapses
        """

        active_minicolumns = torch.Tensor(active_minicolumns)
        apical_input = torch.Tensor(apical_input)

        if apical_growth_candidates is None:
            apical_growth_candidates = apical_input
        apical_growth_candidates = torch.Tensor(apical_growth_candidates)

        self.previous_predicted_cells = self.predicted_cells

        self.activate_cells(
            active_minicolumns=active_minicolumns,
            basal_reinforce_candidates=self.active_cells,
            apical_reinforce_candidates=self.previous_apical_input,
            basal_growth_candidates=self.winner_cells,
            apical_growth_candidates=self.previous_apical_growth_candidates,
            learn=learn
        )

        self.depolarize_cells(
            basal_input=self.active_cells, 
            apical_input=apical_input, 
            learn=learn
        )

        self.previous_apical_input = apical_input.clone()
        self.previous_apical_growth_candidates = apical_growth_candidates.clone()

    def get_predicted_cells(self):
        """
        return prediction from previous timestep.
        """

        return self.previous_predicted_cells
    
    def get_next_predicted_cells(self):
        """
        return prediction for next timestep.
        """

        return self.predicted_cells
    
    def get_next_basal_predicted_cells(self):
        """
        return cells with active basal segments.
        """

        return torch.unique(
            self.map_basal_segments_to_cells(self.active_basal_segments)
        )
    
    def get_next_apical_predicted_cells(self):
        """
        return cells with active apical segments.
        """

        return torch.unique(
            self.map_apical_segments_to_cells(self.active_apical_segments)
        )









    
        

