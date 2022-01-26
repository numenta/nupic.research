# Columns

This package contains algorithms and experiments related to the papers [Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex](https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full), [The HTM Spatial Pooler—A Neocortical Algorithm for Online Sparse Distributed Coding](https://www.frontiersin.org/articles/10.3389/fncom.2017.00111/full), [A Theory of How Columns in the Neocortex Enable Learning the Structure of the World](https://www.frontiersin.org/articles/10.3389/fncir.2017.00081/full), [Locations in the Neocortex: A Theory of Sensorimotor Object Recognition Using Cortical Grid Cells](https://www.frontiersin.org/articles/10.3389/fncir.2019.00022/full). Note that working code for those papers is available in [htmpapers](https://github.com/numenta/htmpapers).

## Introduction to the code

These experiments and algorithms import custom C++ code from `nupic.bindings`. The algorithms are written in two different ways:

- Python algorithms with C++ data structures
- C++ algorithms, C++ data structures

An example comparison: `nupic.research.frameworks.columns.ApicalTiebreakSequenceMemory` vs. `nupic.bindings.algorithms.ApicalTiebreakSequenceMemory`. These are two implementations of the same algorithm. The Python version uses our C++ `nupic.bindings.math.SparseMatrixConnections` data structure to do vectorized operations, and the algorithm is written in Python. The C++ version is faster, partly because it avoids Python overhead, but especially because it can be fast without needing vectorized operations, so it is much more memory efficient. Often the C++ code is actually easier to read than the Python code, due to the fact that the Python code is written in a vectorized style.

### Data structures

The core data structures used by our Python code are the `SparseMatrix` and `SparseBinaryMatrix`. We extend these to add dendritic segments with two different approaches. The `SegmentsSparseMatrix` adds a thin layer to the `SparseMatrix`, and continues using explicit (but relatively cryptic) method names like `rightVecSumAtNZ`. The `SparseMatrixConnections` puts a thicker layer over the `SparseMatrix` and uses more friendly (but not as explicit) names like `computeActivity`.

A tutorial on the SparseMatrix is available [here](https://github.com/numenta/nupic.research.core/blob/master/examples/bindings/sparse_matrix_how_to.py).
