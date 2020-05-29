Introduction
==============

This repository contains the code for experimental algorithm work done internally at Numenta. A description of our core neuroscience research is [available here](https://www.numenta.com/neuroscience-research/). A description of how we are applying neuroscience research to machine learning (our current focus) is [available here](https://numenta.com/machine-intelligence-research).

Open Research
==============

We are dramatically open with our research. We even release our day to day research code into this repository. It contains experimental algorithm code done internally at Numenta. It includes prototypes and experiments with different algorithm implementations.

Please see this [blog entry](https://numenta.com/blog/2018/10/22/framework_for_intelligence_commitment_to_open_science/) for a discussion on our commitment to Open Science and Open Research.

The NuPIC open source community continues to maintain and improve our more stable older algorithms. See https://discourse.numenta.org for discussions on that codebase - you can also post your research related questions there.

The ideas in this repository are constantly in flux as we tweak and experiment. Hence the following DISCLAIMERS:
 
What you should understand about this repository
================================================

- the code can change quickly and without warning as experiments are discarded and recreated
- code will not be production-quality, bug free, or well documented
- if we do work with external partners, that work will probably NOT be here
- we might decide at some point to not do our research in the open anymore and instead delete the whole repository

Papers
======

A complete list of our papers is [available here](https://numenta.com/neuroscience-research/research-publications/). If you are interested in scripts and code used in published papers, [this repository](https://github.com/numenta/htmpapers) contains reproducible code for selected Numenta papers.

Installation
============

OK, enough caveats. Here are some installation instructions though mostly you are on your own. (Wait, was that another caveat?)

When using anaconda virtual environment all you need to do is run the following command and conda will install everything for you. See [environment.yml](./environment.yml):

    conda env create

Otherwise you need can install using setup.py like any python project. Since the contents here change often, we highly recommend installing as follows:
    
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    python setup.py develop

You can test your installation by running the test script from the repository root:

    python setup.py test
  

Archive
=======

Some of our old research code and experiments are archived in the following repositories: 
 
* https://github.com/numenta/htmresearch (requires python 2.7)
* https://github.com/numenta-archive/htmresearch (requires python 2.7)

