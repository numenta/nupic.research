:::bash

.PHONY: clean all

all: figs/hyperparameter_search_panel.png figs/hyperparameter_search_panel_along_tasks.png \
	figs/hyperparameter_search_panel_with_si_dendrites.png

data_hyperparameter_search/segment_search_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py segment_search -n segment_search -o lasttask && \
	mv segment_search_lasttask.pkl segment_search_lasttask.csv segment_search_lasttask_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/segment_search_50_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py segment_search_50 -n segment_search_50 -o lasttask && \
	mv segment_search_50_lasttask.pkl segment_search_50_lasttask.csv segment_search_50_lasttask_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/kw_sparsity_search_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py kw_sparsity_search -n kw_sparsity_search -o lasttask && \
	mv kw_sparsity_search_lasttask.pkl kw_sparsity_search_lasttask.csv kw_sparsity_search_lasttask_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/kw_sparsity_search_50_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py kw_sparsity_search_50 -n kw_sparsity_search_50 -o lasttask && \
	mv kw_sparsity_search_50_lasttask.pkl kw_sparsity_search_50_lasttask.csv kw_sparsity_search_50_lasttask_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/w_sparsity_search_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py w_sparsity_search -n w_sparsity_search -o lasttask && \
	mv w_sparsity_search_lasttask.pkl w_sparsity_search_lasttask.csv w_sparsity_search_lasttask_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/w_sparsity_search_50_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py w_sparsity_search_50 -n w_sparsity_search_50 -o lasttask && \
	mv w_sparsity_search_50_lasttask.pkl w_sparsity_search_50_lasttask.csv w_sparsity_search_50_lasttask_analysis.csv data_hyperparameter_search/

figs/hyperparameter_search_panel.png: hyperparameters_figures.py \
	data_hyperparameter_search/segment_search_lasttask.csv \
	data_hyperparameter_search/segment_search_50_lasttask.csv \
	data_hyperparameter_search/kw_sparsity_search_lasttask.csv \
	data_hyperparameter_search/kw_sparsity_search_50_lasttask.csv \
	data_hyperparameter_search/w_sparsity_search_lasttask.csv \
	data_hyperparameter_search/w_sparsity_search_50_lasttask.csv
	python hyperparameters_figures.py

data_hyperparameter_search/segment_search_all.csv: ../analyze_results.py
	python ../analyze_results.py segment_search -n segment_search -o all && \
	mv segment_search_all.pkl segment_search_all.csv segment_search_all_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/segment_search_50_all.csv: ../analyze_results.py
	python ../analyze_results.py segment_search_50 -n segment_search_50 -o all && \
	mv segment_search_50_all.pkl segment_search_50_all.csv segment_search_50_all_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/kw_sparsity_search_all.csv: ../analyze_results.py
	python ../analyze_results.py kw_sparsity_search -n kw_sparsity_search -o all && \
	mv kw_sparsity_search_all.pkl kw_sparsity_search_all.csv kw_sparsity_search_all_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/kw_sparsity_search_50_all.csv: ../analyze_results.py
	python ../analyze_results.py kw_sparsity_search_50 -n kw_sparsity_search_50 -o all && \
	mv kw_sparsity_search_50_all.pkl kw_sparsity_search_50_all.csv kw_sparsity_search_50_all_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/w_sparsity_search_all.csv: ../analyze_results.py
	python ../analyze_results.py w_sparsity_search -n w_sparsity_search -o all && \
	mv w_sparsity_search_all.pkl w_sparsity_search_all.csv w_sparsity_search_all_analysis.csv data_hyperparameter_search/

data_hyperparameter_search/w_sparsity_search_50_all.csv: ../analyze_results.py
	python ../analyze_results.py w_sparsity_search_50 -n w_sparsity_search_50 -o all && \
	mv w_sparsity_search_50_all.pkl w_sparsity_search_50_all.csv w_sparsity_search_50_all_analysis.csv data_hyperparameter_search/

figs/hyperparameter_search_panel_along_tasks.png: hyperparameters_figures.py \
	data_hyperparameter_search/segment_search_all.csv \
	data_hyperparameter_search/segment_search_50_all.csv \
	data_hyperparameter_search/kw_sparsity_search_all.csv \
	data_hyperparameter_search/kw_sparsity_search_50_all.csv \
	data_hyperparameter_search/w_sparsity_search_all.csv \
	data_hyperparameter_search/w_sparsity_search_50_all.csv
	python hyperparameters_figures.py

data_hyperparameter_search_with_si_dendrites/segment_search_with_si_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py segment_search_with_si -n segment_search_with_si -o lasttask && \
	mv segment_search_with_si_lasttask.pkl segment_search_with_si_lasttask.csv segment_search_with_si_lasttask_analysis.csv data_hyperparameter_search_with_si_dendrites/

data_hyperparameter_search_with_si_dendrites/kw_sparsity_search_with_si_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py kw_sparsity_search_with_si -n kw_sparsity_search_with_si -o lasttask && \
	mv kw_sparsity_search_with_si_lasttask.pkl kw_sparsity_search_with_si_lasttask.csv kw_sparsity_search_with_si_lasttask_analysis.csv data_hyperparameter_search_with_si_dendrites/

data_hyperparameter_search_with_si_dendrites/w_sparsity_search_with_si_lasttask.csv: ../analyze_results.py
	python ../analyze_results.py w_sparsity_search_with_si -n w_sparsity_search_with_si -o lasttask && \
	mv w_sparsity_search_with_si_lasttask.pkl w_sparsity_search_with_si_lasttask.csv w_sparsity_search_with_si_lasttask_analysis.csv data_hyperparameter_search_with_si_dendrites/

figs/hyperparameter_search_panel_with_si_dendrites.png: hyperparameters_dendrites_with_si_figures.py \
	data_hyperparameter_search_with_si_dendrites/segment_search_with_si_lasttask.csv \
	data_hyperparameter_search_with_si_dendrites/kw_sparsity_search_with_si_lasttask.csv \
	data_hyperparameter_search_with_si_dendrites/w_sparsity_search_with_si_lasttask.csv
	python hyperparameters_dendrites_with_si_figures.py

clean:
	rm -rf data_hyperparameter_search/*
	rm -rf data_hyperparameter_search_with_si/*
	rm -rf figs/*
