# Spatial Pooler

This package contains the algorithm related to the papers [The HTM Spatial Pooler—A Neocortical Algorithm for Online Sparse Distributed Coding](https://www.frontiersin.org/articles/10.3389/fncom.2017.00111/full) and [Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex](https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full). Note that working code for those papers is available in [htmpapers](https://github.com/numenta/htmpapers).

## Details about the code

This version of the Spatial Pooler cleans up the prior version of the Spatial Pooler, which had C++ bindings to implement core functionality.
Notably, this algorithm is implemented purely in Python 3 with the NumPy library.
To check the correctness of the algorithm, please refer to the tests in the `tests/` folder.

## Details about the algorithm

This version of the Spatial Pooler removes the functionality for `wrapAround` local inhibition of minicolumns.