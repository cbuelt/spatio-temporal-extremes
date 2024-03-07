# Estimation of Extremes for Spatio-Temporal Processes with Neural Networks

## Abstract

 

## Setup

1. Install `Python` (see [here](https://python.org/)) and `R` (see [here](https://www.r-project.org/)).
	- `Python` is easiest installed by using the anaconda package manager (see [here](https://www.anaconda.com/)).
1. Clone project   
    - ```git clone https://github.com/cbuelt/thesis```
1. Install the `R` packages.
	- Source the file `utils/requirements.R`.
1. Install the python requirements.
    - Create a conda environment using ```conda env create -f utils/requirements.yml```
    - For help see see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Reproducing results

For all models and experiments, the results are saved in the corresponding data folders. The metrics can be recalculated from the model estimations. The estimations itself can be run again, based on the simulated training and testing data. The whole simulations can also be done from scratch, but keep in mind that new data would change the results.

## Repository structure

| Folder | Description |
| ---- | ----------- | 
| `abc` | Contains code related to the Approximate Bayesian Computing (ABC) method. |
| `application` | Contains code regarding the empirical part of the thesis. |
| `data` | Contains the data used in the thesis and required to reproduce the results. |
| `evaluation` | Contains code for evaluating the different models. |
| `mle` | Contains code related to the Pairwise Likelihood approach. |
| `networks` | Contains the implementation of the neural networks. |
| `plots` | Contains code for the plots used in the thesis. |
| `simulation` | Contains code for simulating max-stable processes. |
| `utils` | Contains utility functions. |


