# DFLike

This is a collection of Differentiable Likelihoods for Simons Observatory.

It currently contains a JAX implementation of [LAT_MFLike](https://github.com/simonsobs/LAT_MFLike/) (and parts of [fgspectra](https://github.com/simonsobs/fgspectra) which it relies on), as well a simple Gaussian Lensing likelihood based on the one from [SOLikeT](https://github.com/simonsobs/SOLikeT).

## Features

 - [x] Same foreground model with bandpass integration from ACT DR6 and SO LAT.
 - [x] Chromatic beam correction
 - [x] chi square agreement with ACT DR6 at same cosmology.
 - [x] CMB lensing.
 - [x] Auto-differentiation through JAX.
 - [x] Examples notebooks
 - [ ] Documentation
 - [ ] Examples of gradient MC solvers (e.g. HMC or NUTS) to replicate ACT DR6 cosmology.
