# Tutorial Notebooks

This directory contains a series of notebooks serving as examples to teach people how to use and exploit the differentiable codes shown in here.

The tutorials can be followed out of order, but in general are intended to follow one after the next. They are slightly more verbose and explicit than necessary, and they perform certain actions in place rather than deferring to built-in utility functions.


## 1. Introduction to Likelihoods

The first notebook is intended as a first example to show how the python interface for the code works. To start, the notebook imports and instantiates the Multi-frequency likelihood, and compares it to the official `MFLike` likelihood for the SO LAT.

The notebook shows the general structure and functionality of likelihoods and theory codes, and how to access them directly in python.


## 2. Introduction to Differentiability

This notebook shows how `jax` autodifferentiability works by constructing several functions that exploit differentiability. It shows how to modify functions to keep specific parameters fixed with minimal overhead.


## 3. Cosmology

This notebook utilizes the `cosmopower_jax` wrapper to include CMB cosmopower emulators to propagate derivatives to cosmology.


## 4. Lensing

This notebook shows how to use the Lensing likelihood and theory codes. It also shows how to do quick forecasting of errors for the CMB in combination with Lensing.


## 5. Parameter Biases

This notebook shows how to utilize Fisher forecasting functions to compute posterior approximations, and how to propagate model mismatches into parameter biases.


## 6. Pipelines

This notebook summarizes some of the things shown in previous notebooks, utilizing the builtin `Pipeline` structure to automate a lot of operations.
