# RSM & Related Models

## Quick start (PTB)

* Ensure nupic.torch and nupic.research are installed:
	* `cd ~/nta/nupic.torch && python setup.py develop`
    * `cd ~/nta/nupic.research && python setup.py develop`
* Change to rsm directory
* `pip install -r requirements.txt`
* Generate fasttext embedding (if using): `python ptb/generate_ft_ptb.py` (first adjust PATH in file)
* Run an experiment (e.g. on PTB dataset): `python rsm_ray.py --config=ptb/ptb_experiments.cfg --experiment=[EXP_NAME]`
