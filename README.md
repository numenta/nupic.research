Introduction
==============

This repository contains the code for experimental algorithm work done internally at Numenta. A description of that research is [available here](https://www.numenta.com/neuroscience-research/).

Open Research
==============

We have released all our commercial HTM algorithm code to the open source community within NuPIC. The NuPIC open source community continues to maintain and improve that regularly (see https://discourse.numenta.org for discussions on that codebase. Internally we continue to evolve our theory towards a full blown cortical framework.

We get a lot of questions about it and we wondered whether it is possible to be even more open about that work. Could we release our day to day research code in a public repository? Would people get confused? Would it slow us down?

We decided to go ahead and create nupic.research. It contains experimental algorithm code done internally at Numenta. The code includes prototypes and experiments with different algorithm implementations.

Our research ideas are constantly in flux as we tweak and experiment. This is all temporary, ever-changing experimental code, which poses some challenges. Hence the following DISCLAIMERS:

 
What you should understand about this repository
================================================

- the code can change quickly and without warning as experiments are discarded and recreated
- code will not be production-quality, buggy, or well documented
- if we do work with external partners, that work will probably NOT be here
- we might decide at some point to not do our research in the open anymore and instead delete the whole repository


Installation
============

OK, enough caveats. Here are some installation instructions though mostly you are on your own. (Wait, was that another caveat?)

Install using setup.py like any python project. Since the contents here change often, we highly recommend installing as follows:

    python setup.py develop


You can test your installation by running the test script from the repository root:

    python setup.py test
  

Archive
=======

Some of our old research code and experiments are archived in the following repositories: 
 
* https://github.com/numenta/htmresearch (requires python 2.7)
* https://github.com/numenta-archive/htmresearch (requires python 2.7)

