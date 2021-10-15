# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

"""
A simple implementation of dendrite weights. This combines the output from a (sparse)
linear layer with the output from a set of dendritic segments.
"""
import abc

import torch

from nupic.torch.modules.sparse_weights import SparseWeights, SparseWeights2d

from .apply_dendrites import (
    DendriticAbsoluteMaxGate1d,
    DendriticAbsoluteMaxGate2d,
    DendriticBias1d,
    DendriticGate1d,
    DendriticGate2d,
)
from .dendrite_segments import DendriteSegments


class DendriticLayerBase(SparseWeights, metaclass=abc.ABCMeta):
    """
    Base class for all Dendritic Layer modules.

    This combines a DendriteSegments module with a SparseLinear module.
    The output from the dendrite segments (shape of num_units x num_segments)
    is applied to the output of of the linear weights (shape of num_units).
    Thus, each linear output unit gets modulated by a set of dendritic segments.
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        TODO: specify the type - what is module_sparsity type?
        :param module: linear module from in-units to out-units
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.dim_context = dim_context
        self.segments = None
        super().__init__(
            module,
            sparsity=module_sparsity,
            allow_extremes=True
        )

        self.segments = DendriteSegments(
            num_units=module.weight.shape[0],
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context):
        """Compute of linear layer and apply output of dendrite segments."""
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)

    @property
    def segment_weights(self):
        return self.segments.weights


class BiasingDendriticLayer(DendriticLayerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_bias = DendriticBias1d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a bias."""
        return self.dendritic_bias(y, dendrite_activations).values


class GatingDendriticLayer(DendriticLayerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_gate = DendriticGate1d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        return self.dendritic_gate(y, dendrite_activations).values


class AbsoluteMaxGatingDendriticLayer(DendriticLayerBase):
    """
    This layer is similar to `GatingDendriticLayer`, but selects dendrite activations
    based on absolute max activation values instead of just max activation values. For
    example, if choosing between activations -7.4, and 6.5 for a particular unit, -7.4
    will be chosen, and its sign will be kept.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_absolute_max_gate = DendriticAbsoluteMaxGate1d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        return self.dendritic_absolute_max_gate(y, dendrite_activations).values


class OneSegmentDendriticLayer(SparseWeights):
    """
    Class for a layer of units with exactly one sparse dendritic segment per unit. With
    this assumption the segments are just a straightforward linear SparseWeights layer.
    It seems to be 3-6 times faster than other implementations depending on settings.
    """

    def __init__(
        self, module, dim_context, module_sparsity, dendrite_sparsity,
        num_segments=1, dendrite_bias=False
    ):
        """
        :param module: linear module from in-units to out-units
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param num_segments: number of dendrite segments per out-unit. Must be 1.
        :param dendrite_bias: bool indicating whether or not dendrite activations have
               an additive bias
        """
        assert(num_segments == 1)

        self.dim_context = dim_context
        self.segments = None

        super().__init__(module,
                         sparsity=module_sparsity,
                         allow_extremes=True)

        self.segments = SparseWeights(
            torch.nn.Linear(dim_context,
                            module.weight.shape[0],
                            bias=dendrite_bias),
            sparsity=dendrite_sparsity,
            allow_extremes=True
        )

        self.rezero_weights()

    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    def forward(self, x, context):
        """Compute of linear layer and apply output of dendrite segments."""
        y = super().forward(x)
        dendrite_activations = self.segments(context)
        return self.apply_dendrites(y, dendrite_activations)

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a sigmoidal gating mechanism."""
        return y * torch.sigmoid(dendrite_activations)

    @property
    def segment_weights(self):
        return self.segments.module.weight


class ZeroSegmentDendriticLayer(SparseWeights):
    """
    Class for a layer of units with no dendritic segments per unit. This is identical
    to a normal feed-forward layer, but useful for debugging to ensure we use the same
    code paths and that everything else is identical.
    """

    def __init__(
        self, module, dim_context, module_sparsity, dendrite_sparsity,
        num_segments=0, dendrite_bias=False
    ):
        """
        :param module: linear module from in-units to out-units
        :param dim_context: (ignored)
        :param module_sparsity: sparsity applied over linear module
        :param dendrite_sparsity: (ignored)
        :param num_segments: number of dendrite segments per out-unit. Must be 0.
        :param dendrite_bias: (ignored)
        """
        assert(num_segments == 0)

        self.dim_context = 0
        super().__init__(module,
                         sparsity=module_sparsity,
                         allow_extremes=True)

        self.segments = None
        self.rezero_weights()

    def forward(self, x, context):
        return super().forward(x)

    @property
    def segment_weights(self):
        return None


class ZeroSegmentDendriticLayerCatContext(SparseWeights):

    """
    Just like ZeroSegmentDendriticLayer, but takes insted of ignoring context, it
    concatenates the context to the input features. Context is assumed to be 1-hot.
    Main use case is comparing an MLP to dendritic network. Dendritic networks have
    context by default, so this is a way of manufacturing a type of context for MLPs.
    """

    def __init__(
        self, module, dim_context, module_sparsity, dendrite_sparsity,
        num_segments=0, dendrite_bias=False
    ):
        """
        :param module: linear module from in-units to out-units
        :param dim_context: must be exactly the same as the number of tasks
        :param module_sparsity: sparsity applied over linear module
        :param dendrite_sparsity: (ignored)
        :param num_segments: number of dendrite segments per out-unit. Must be 0.
        :param dendrite_bias: (ignored)
        """
        assert(num_segments == 0)

        self.dim_context = 0
        super().__init__(module,
                         sparsity=module_sparsity,
                         allow_extremes=True)

        self.segments = None
        self.rezero_weights()

    def forward(self, x, context):
        x = torch.cat((x, context))
        return super().forward(x)

    @property
    def segment_weights(self):
        return None



class DendriticLayer2dBase(SparseWeights2d, metaclass=abc.ABCMeta):
    """
    Base class for all 2d Dendritic Layer modules.

    Similar to the DendriticLayerBase class, the output from the dendrite segments
    is applied to the output of each channel. Thus, each channel output gets
    modulated by a set of dendritic segments.
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        :param module: conv2d module which performs the forward pass
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.segments = None
        super().__init__(module, sparsity=module_sparsity)

        self.segments = DendriteSegments(
            num_units=module.out_channels,
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context):
        """
        Computes the forward pass through the `torch.nn.Conv2d` module and applies the
        output of the dendrite segments.
        """
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)


class GatingDendriticLayer2d(DendriticLayer2dBase):
    """
    A convolutional version of `GatingDendriticLayer`. The multiplicative dendrite
    outputs are applied element-wise to each output channel. That is, for a given
    output channel, all activation values (determined by the convolution operation) are
    multiplied by a single value computed via dendrites.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_gate = DendriticGate2d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        return self.dendritic_gate(y, dendrite_activations).values


class AbsoluteMaxGatingDendriticLayer2d(DendriticLayer2dBase):
    """
    A convolutional version of `AbsoluteMaxGatingDendriticLayer`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_absolute_max_gate = DendriticAbsoluteMaxGate2d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        return self.dendritic_absolute_max_gate(y, dendrite_activations).values
