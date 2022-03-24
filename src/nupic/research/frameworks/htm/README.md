# HTM Algorithms

## Spatial Pooler

The `spatial_pooler.py` file contains the algorithm related to the papers [The HTM Spatial Pooler—A Neocortical Algorithm for Online Sparse Distributed Coding](https://www.frontiersin.org/articles/10.3389/fncom.2017.00111/full) and [Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex](https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full). Note that working code for those papers is available in [htmpapers](https://github.com/numenta/htmpapers).

### Details about the code

This version of the Spatial Pooler cleans up the prior version of the Spatial Pooler, which had C++ bindings to implement core functionality.
Notably, this algorithm is implemented purely in Python 3 with the NumPy library.
To check the correctness of the algorithm, please refer to the tests in the `tests/` folder.

### Details about the algorithm

This version of the Spatial Pooler removes the functionality for `wrapAround` local inhibition of minicolumns.

## Temporal Memory

The `temporal_memory/temporal_memory_apical_tiebreak.py` file contains the algorithm related to the papers [Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex](https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full), [A Theory of How Columns in the Neocortex Enable Learning the Structure of the World](https://www.frontiersin.org/articles/10.3389/fncir.2017.00081/full), and [Locations in the Neocortex: A Theory of Sensorimotor Object Recognition Using Cortical Grid Cells](https://www.frontiersin.org/articles/10.3389/fncir.2019.00022/full). Note that working code for those papers is available in [htmpapers](https://github.com/numenta/htmpapers).

### Details about the code

This version of the Temporal Memory is written purely in PyTorch. Extensions of this code can be found in the `temporal_memory` folder, like sequence memory or pair memory. Full disclaimer: although this version of Temporal Memory is correct, it is significantly slower than the [Python version with C++ bindings](https://github.com/numenta/nupic.research/tree/master/packages/columns). Please use this version for debugging or quick prototyping. Production use-cases should refer to the Python version with C++ bindings.

### Details about the algorithm

Like the Python version with C++ bindings, this version does not delete basal or apical segments if they are inactive.
