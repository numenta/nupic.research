# RSM & Related Models

## Quick start (PTB)

* Change to rsm directory
* `pip install -r requirements.txt`
* Generate fasttext embedding (if using): `python ptb/generate_ft_ptb.py` (first adjust PATH in file)
* Run an experiment (e.g. on PTB dataset): `python rsm_ray.py --config=ptb/ptb_experiments.cfg --experiment=[EXP_NAME]`