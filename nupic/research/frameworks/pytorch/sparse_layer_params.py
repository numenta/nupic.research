# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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


def auto_sparse_conv_params(in_channels, out_channels, kernel_size):
    """
    Given conv2d parameters, automatically calculate sparsity parameters
    for `SparseWeights2d` module.

    Note:
    This is highly experimental and likely to change.

    :return: an instance of LayerParams
    """
    weights_per_channel = kernel_size * kernel_size * in_channels
    if weights_per_channel < 100:
        weights_density = 0.75

    elif weights_per_channel < 200:
        weights_density = 0.5

    elif weights_per_channel < 500:
        weights_density = 0.4

    elif weights_per_channel < 1000:
        weights_density = 0.3

    elif weights_per_channel < 2000:
        weights_density = 0.2

    elif weights_per_channel < 4000:
        weights_density = 0.2

    else:
        weights_density = 0.2

    return dict(
        weight_sparsity=weights_density,
    )


def auto_sparse_activation_params(in_channels, out_channels, kernel_size):
    """
    Given conv2d parameters, automatically calculate sparsity parameters
    for the `KWinners2d` activation layer to follow.

    Note:
    This is highly experimental and likely to change.

    :return: a dict to pass to `KWinners2d` as params.
    """

    if out_channels > 64:
        percent_on = 0.2
    else:
        percent_on = 0.3

    return dict(
        percent_on=percent_on,
        boost_strength=1.5,
        boost_strength_factor=0.95,
        local=True,
    )


class LayerParams(object):
    """
    Class to manage a layer's parameters.

    Specifically, this class helps one specify separate parameters for
    convolutional, linear, and activation layers. For each respective
    layer typer, either of the following may be specified
    1) a function which takes arbitrary inputs and outputs a dictionary, or
    2) a predefined dictionary.

    Accordingly, the former (1) of the two takes precedent
    when determining the desired set of parameters and both 1 and 2
    are specified. If neither are specified, then an empty dictionary is returned.
    """

    def __init__(
        self,
        default_linear_params=None,
        default_conv_params=None,
        default_activation_params=None,
        linear_params_func=None,
        conv_params_func=None,
        activation_params_func=None,
    ):
        self.default_linear_params = default_linear_params
        self.default_conv_params = default_conv_params
        self.default_activation_params = default_activation_params
        self.linear_params_func = linear_params_func
        self.conv_params_func = conv_params_func
        self.activation_params_func = activation_params_func

    def __repr__(self):

        lin_source = self.linear_params_func or self.default_linear_params
        conv_source = self.conv_params_func or self.default_conv_params
        act_source = self.activation_params_func or self.default_activation_params
        return self.__class__.__name__ + \
            "(\n\tlinear_params={},\n\tconv_params={},\n\tactivation_params={})".format(
                repr(lin_source), repr(conv_source), repr(act_source)
            )

    def get_linear_params(self, *args, **params):

        if self.linear_params_func:
            return self.linear_params_func(*args, **params)
        elif self.default_linear_params:
            return self.default_linear_params
        else:
            return {}

    def get_conv_params(self, *args, **params):

        if self.conv_params_func:
            return self.conv_params_func(*args, **params)
        elif self.default_conv_params:
            return self.default_conv_params
        else:
            return {}

    def get_activation_params(self, *args, **params):

        if self.activation_params_func:
            return self.activation_params_func(*args, **params)
        elif self.default_activation_params:
            return self.default_activation_params
        else:
            return {}


class LayerParamsByKeys(LayerParams):
    """
    Class for creating easier to handle instances of `LayerParams` for
    specific cases. For instance, sub-classes may be handled as

    Example 1:
    ```
    p = MyParams(a=2, b=3, c=2)
    print(p)
    >>> MyParams(
        linear_params={'a':2},
        conv_params={},
        activation_params={'b':3, 'c':2})
    ```
    where the mapping between a, b, and c, to linear, conv, or activation params
    is known from the user-specification of MyParams. Such definitions with
    this class are relatively straightforward.

    Example 2:
    ```
    class MyParams(LayerParamsByKeys):
        linear_params_keys = ["a"]
        activation_params = ["b", "c"]
    ```

    This example creates the params as seen above in Example 1.
    We may also specify default arguments and alternative key-words
    for saving a our params.

    Example 3:
    ```
    class OtherParams(LayerParamsByKeys):

        linear_params_keys = [
            (
                "linear_sparisty",  # <key name to pass to OtherParams.__init__>
                0.5                 # <default_value>
            )
        ]
        conv_params_keys = [
            (
                "conv_sparsity",   # <key name to pass to OtherParams.__init__>
                0.1,               # <default_value>
                "weight_sparsity"  # "<target name for saving to `default_conv_params`>"
            )
        ]
        activation_params_keys = [
            (
                "activation_sparsity",  # <key name to pass to OtherParams.__init__>
                ...,                    # `...` => No Default
                "percent_on"            # "<target name for saving>"
            )
        ]

    p = OtherParams(conv_sparsity=0.7)
    print(p)
    >>> OtherParams(
        linear_params={"linear_sparisty": 0.5},
        conv_params={"weight_sparsity": 0.7},
        activation_params={}
    )
    ```

    In the above example, notice how "linear_sparisty" gets a default value despite not
    being passed to `OtherParams.__init__`. Also, notice how "weight_sparsity" gets
    saved for the `conv_params` but it's passed to `OtherParams.__init__` as
    `conv_sparsity`.

    This kind of behavior is useful when the params for the linear, conv, or,
    activations layers share the same name. For example, both `SparseWeights` and
    `SparseWeights2d` take a `weight_sparsity` param, so it may help to pass them
    as `lin_sparsity` and `conv_sparsity`, yet save them both as `weight_sparsity`
    in their respective parameter dictionaries.

    Finally, note that a `activation_params` remains empty in Example 3.
    Despite have an entry for the default value, the value of `...` is treated
    as if no default has been specified. This way, one may specify a target-name
    for saving without having to specify a default value as well.
    """

    def __init__(
        self,
        linear_params_func=None,
        conv_params_func=None,
        activation_params_func=None,
        **kwargs,
    ):

        self.linear_params_func = linear_params_func
        self.conv_params_func = conv_params_func
        self.activation_params_func = activation_params_func

        if hasattr(self, "linear_params_keys"):
            self.default_linear_params = self._get_values_from_dict(
                kwargs, self.linear_params_keys
            )
        else:
            self.default_linear_params = None

        if hasattr(self, "conv_params_keys"):
            self.default_conv_params = self._get_values_from_dict(
                kwargs, self.conv_params_keys
            )
        else:
            self.default_conv_params = None

        if hasattr(self, "activation_params_keys"):
            self.default_activation_params = self._get_values_from_dict(
                kwargs, self.activation_params_keys
            )
        else:
            self.default_activation_params = None

    def _get_values_from_dict(self, d, keys):
        """
        Creates dict of key value pairs collected from the specified 'keys' of 'd'.

        Note that if a key is presented as a tuple (e.g. ("some_key", <value>)), then
        the first value will be interpreted as the key, and the second will be
        interpreted as the default value. Additionally, any third value of the tuple
        will be interpreted as a the `target_name` for the key. That is, although
        `key in d` may be true, it will be saved as `key_value[target_name] = value`.
        """
        key_values = {}
        for key in keys:

            if isinstance(key, tuple):
                key, default_value, target_name = (key + (None, ) * 3)[:3]
            else:
                target_name = None
                default_value = None

            target_name = target_name or key
            if key in d:
                key_values[target_name] = d[key]
            elif default_value is not ...:
                key_values[target_name] = default_value

        return key_values


class SpareWeightsLayerParams(LayerParamsByKeys):
    """
    LayerParams class for specifying parameters to
    `SparseWeights` (linear params), `SparseWeights2d` (conv params)
    and `KWinners2d` (activation params).

    Example:
    ```
        sw = SpareWeightsLayerParams(
        percent_on=0.3,
        boost_strength=1.2,
        boost_strength_factor=1.0,
        local=False,
        linear_weight_sparsity=0.3,
        conv_params_func=auto_sparse_conv_params,
    )

    print(sw)
    >>> SpareWeightsLayerParams(
            linear_params={'weight_sparsity': 0.3},
            conv_params=<function auto_sparse_conv_params at 0x10b4a1830>,
            activation_params={
                'percent_on': 0.3,
                'boost_strength': 1.2,
                'boost_strength_factor': 1.0,
                'local': False
            }
        )
    """

    linear_params_keys = [
        ("linear_weight_sparsity", ..., "weight_sparsity")  # ... => no default
    ]
    conv_params_keys = [
        ("conv_weight_sparsity", ..., "weight_sparsity")  # ... => no default
    ]
    activation_params_keys = [
        "percent_on",
        "boost_strength",
        "boost_strength_factor",
        "k_inference_factor",
        "local",
    ]
