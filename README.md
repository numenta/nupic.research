Introduction
==============

This repository contains the code for experimental and neuroscience algorithm work done internally at Numenta. A description of our core neuroscience research is [available here](https://www.numenta.com/neuroscience-research/). A description of how we are applying neuroscience research to machine learning (our current focus) is [available here](https://www.numenta.com/technology/).

Open Research
==============

This repository contains much of our basic code for experimental algorithm and neuroscience theories done internally at Numenta. It includes prototypes and experiments of several ideas behind the Thousand Brains Theory, sometimes with several different implementations of the same algorithm. Many of our related internal research meetings are recorded and [released on YouTube](https://www.youtube.com/c/NumentaTheory/videos).

The NuPIC open source community continues to maintain and improve our more stable older algorithms. See https://discourse.numenta.org for discussions on that codebase - you can also post your research related questions there.

The ideas in this repository are constantly in flux as we tweak and experiment. Anyone looking through this should understand the following DISCLAIMERS:
 
What you should understand about this repository
================================================

- the code can change quickly and without warning as experiments are discarded and recreated
- code will not be production-quality, bug free, or well documented
- if we do work with, or for, external partners and customers, that work will probably NOT be here
- we might decide at some point to not do our research in the open anymore and instead delete the whole repository

Papers
======

A list of our papers is [available here](https://numenta.com/neuroscience-research/research-publications/). If you are interested in the scripts and code used in published papers, [this repository](https://github.com/numenta/htmpapers) contains reproducible code for selected Numenta papers.

Installation
============

OK, enough caveats. Here are some installation instructions though mostly you are on your own. (Wait, was that another caveat?)

When using anaconda virtual environment all you need to do is run the following command and conda will install everything for you. See [environment.yml](./environment.yml):

    conda env create

You can test your installation by running the test script from the repository root:

    pytest

Active Projects
=======

Some of our active research code and experiments can also be found in the following repositories:

* https://github.com/numenta/nupic.embodied
  
Archive
=======

Some of our old research code and experiments are archived in the following repositories: 
 
* https://github.com/numenta/htmresearch (requires python 2.7)
* https://github.com/numenta-archive/htmresearch (requires python 2.7)

