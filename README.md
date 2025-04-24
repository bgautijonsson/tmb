# Spatial GEV Model with TMB

## Overview

This project implements a spatial Generalized Extreme Value (GEV) model for environmental data observed at multiple locations. The goal is to perform simultaneous maximum likelihood estimation of GEV parameters for thousands of stations, potentially incorporating complex dependence structures via Gaussian copulas or other spatial correlation models.

The Template Model Builder (TMB) framework is used for automatic differentiation and efficient optimization. TMB allows us to write models in C++ and perform parameter estimation in R, combining both high performance and convenient workflows.

## Motivation

Modeling environmental extremes (e.g., rainfall, temperature, wind speeds) often involves fitting GEV distributions to data from many locations. Traditional station-by-station estimation is straightforward but fails to borrow strength across locations. A spatial model allows improved inference and smoother parameter fields, giving more stable estimates and better uncertainty quantification. By integrating TMB, we can:

- Easily define and differentiate complex likelihoods.
- Scale up to tens of thousands of parameters.
- Integrate spatial dependence structures (e.g., via Gaussian copulas) without manually deriving gradients and Hessians.

## Data

- **Input Data:** A matrix of observations with dimensions \(N \times P\), where \(N\) is the number of observations (e.g., annual maxima) per station, and \(P\) is the number of stations.
- **Parameters:** For each station, we estimate GEV parameters \((\psi, \tau, \phi)\) or directly \((\mu, \sigma, \xi)\) through a transformation. Extensions may include global parameters, spatial random effects, or latent fields.

## Model Description

### GEV Parameterization

We use a parameterization that improves stability and ensures positivity where necessary. For example:

- \(\mu = \exp(\psi)\) to ensure \(\mu > 0\).
- \(\sigma = \exp(\psi + \tau)\) to ensure \(\sigma > 0\).
- \(\xi = (1 + \exp(-\phi))^{-1}\) to ensure \(\xi \in (0,1)\) (or adapt transformations as needed).

Each station \(p\) has parameters \((\psi_p, \tau_p, \phi_p)\). The joint log-likelihood under independence is the sum of station-level log-likelihoods. When introducing a spatial structure (e.g., a Gaussian copula), the likelihood involves the joint distribution of transformed data across stations.

### Spatial Dependence (Future Steps)

Initially, the model may treat stations as independent. Later, we will introduce a Gaussian copula or Gaussian random field. This will entail defining a correlation matrix (possibly based on a Matérn covariance function) and linking station-level parameters or latent fields to this structure.

### Priors or Regularization

For improved stability and to facilitate smoothing after the MLE step, we may include priors on the parameters or treat them as random effects. TMB can handle random effects efficiently via Laplace approximations.
