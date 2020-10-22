# Prognostics Algorithm Python Package

The Prognostic Algorithm Package is a python framework for model-based prognostics (computation of remaining useful life) of engineering systems, and provides a set of algorithms for state estimation and prediction, including uncertainty propagation. The algorithms take as inputs prognostic models (from NASA's Prognostics Model Package), and perform estimation and prediction functions. The library allows the rapid development of prognostics solutions for given models of components and systems. Different algorithms can be easily swapped to do comparative studies and evaluations of different algorithms to select the best for the application at hand.

## Installation
1. Use `pip install -r requirements.txt` to install dependencies 
2. Ensure `prog_models` package is in path. This could be done by moving the `prog_models` directory into the path, or by adding the path to the `prog_models` package 

## Directory Structure 

`prog_algs/` - The prognostics algorithm python package <br>
 |-`predictors/` - Algorithms for performing the prediction step of model-based prognostics <br>
 |-`samplers/` - Standard tools for performing state sampling <br>
 |-`state_estimators/` - Algorithms for performing the state estimation step of model-based prognostics <br>
`example.py` - An example python script using prog_algs <br>
`README.md` - The readme (this file)
`requirements.txt` - python library dependiencies required to be met to use this package. Install using `pip install -r requirements.txt`

## Citing this repository
Use the following to cite this repository:

```
@misc{2020_nasa_prog_model,
    author    = {Christopher Teubert and Chetan Kulkarni},
    title     = {Prognostics Algorithm Python Package},
    month     = Oct,
    year      = 2020,
    version   = {0.0.1},
    url       = {TBD}
    }
```

The corresponding reference should look like this:

C. Teubert, and C. Kulkarni, Prognostics Algorithm Python Package, v0.0.1, Oct. 2020. URL TBD.