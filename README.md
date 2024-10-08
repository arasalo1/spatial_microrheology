# Probabilistic analysis of spatial viscoelastic cues in 3D cell culture using magnetic microrheometry

## Data
Download data from https://osf.io/eakf2/ and extract to ```data/```
## Requirements

Install cmdstanpy https://mc-stan.org/cmdstanpy/installation.html

`pip install -r requirements.txt`

Install Jax for the calibration and posterior field generation
https://jax.readthedocs.io/en/latest/installation.html

## Structure
* ```field_generation.ipynb``` shows how the posterior viscoelastic fields can be calculated. Replicates the gradient picture
* ```spatial.ipynb``` samples all the posteriors
* ```results.ipynb``` shows how the 
* ```calibration.ipynb``` defines the calibration model
* ```models/``` defines the spatial models