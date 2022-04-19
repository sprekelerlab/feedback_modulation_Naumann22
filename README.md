# Code for Naumann et al., 2022 (eLife)

Here you can find the code for the main findings of the publication "Invariant neural subspaces maintained by feedback modulation" by LB Naumann, J Keijser and H Sprekeler (eLife, 2022).

This repository contains:
* a script containing all models and training/testing functions
* a script to generate data (sources and contexts)
* 3 demo scripts to reproduce the main findings from the paper
* utility scripts to helps with analysis and plotting

## Prerequisites

The simulations were run in `python 3.7.6` on macOS, but also tested on Linux.

1. You need to install `pytorch`, follow the instructions on https://pytorch.org/
2. You also need `numpy`, `scipy`, `scikit-learn`, `tensorboard`, `matplotlib`

The package versions used to run the simulations in the paper can be found in `package-versions.txt`. 

## Model script
The script `model_base.py` contains the code for all models trained and tested for the paper. Each model variant is a separate class.
`UnmixingModel` is the parent class. The model variants are:
* `ModulationNet` : the simplest model variant, used in Fig. 1 and 2
* `SpatialNet` : model variant with spatially diffuse modulation, use in Fig. 3-5
* `DaleRateNet` : model with biological constraints such as positive rates and Dale's law, use in Fig. 6 and 7

In addition to the models, this script containes a `Runner` class with a training and and a testing function that can be used with all model classes.

The model script also contains code in the bottom illustrating how the models can be trained and tested. More specific examples can be found in the demo scripts (see below).

## Data generation script
The script `data_maker.py` contains functions to generate source signals, contexts and wrap the contexts into training and testing data for the `Runner` (see above).

## Demo scripts

To reproduce the core findings of the paper, you can run the demo scripts. The demo scripts can be used to train and/or load and test the models.
**WARNING:** Training a model can take several hours!
The demo scripts contain flags `LOAD_MODEL` and `SAVE_MODEL` that can be used to save and load trained models.
It is highly recommended to first train a model, save it and then load the results from the saved model to play with it and plot results.
The models should be trained for at least 3000 batches, but results get better for 5000-10000 batches.
You can change the numner of batches using the variable `n_batch`.

If you do not want to train the models from scratch you can just load the pre-trained models by setting `n_batch=10000` and `LOAD_MODEL=True`.
The script will then load a trained PyTorch model from the `trained_models` directory.

There are 3 demos:

1. `demo_proof_of_concept.py` reproduces the main findings of Fig. 1, but can also be used to train models with slower feedback or for different input configurations as in Fig. 2.
2. `demo_spatial_net.py` trains a model with spatially diffuse and slow modulation as for the default parameters in Fig. 3.
3. `demo_invariant_subspace.py` reproduces the main findings of Fig. 4 and 5, revealing invariant subspaces.

Note that the demo scripts do not contain any parameter explorations because this would require to train multiple models on different parametrisations.
As training a single model takes several hours, this would be a painful endeavour that you do not want to perform on a single core.
To test the models' behaviour for a range of parameters, it is recommended to use a compute cluster to run several parametrisations in parallel.
The results for the paper were also obtained from a compute cluster.

## Helper scripts
The two scripts `analyses.py` and `helpers.py` contain functions for data analysis and some helpers for plotting.

If you have any further questions about the models or the code, feel free to contact the first author Laura Bella Naumann. You can find the e-mail address on the published eLife paper.
