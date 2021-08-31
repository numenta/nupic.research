These files
- `hyperparameters_figures.py`
- `cns2021_figure1c.py`

are able to generate summary figures of hyperparameter search in the config file `../experiments/hyperparameter_search.py`

To reproduce the figures:

1. Run the config file `hyperparameter_search.py` to generate the data. You need to run the following configs:

  - segment_search
  - kw_sparsity_search
  - w_sparsity_search
  - segment_search_50
  - kw_sparsity_search_50
  - w_sparsity_search_50
  - segment_search_with_si
  - kw_sparsity_search_with_si
  - w_sparsity_search_with_si

2. Once the data has been generated you just need to run the makefile using `make`. You will need to run `make` twice. The first time will error out in the middle but that is expected (it's due to the fact that some of the plots are in the same python script and not in two separated files).

Once you are done use `make clean` to delete the generated csv, pkl, and png files.
