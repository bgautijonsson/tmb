# R code
library(tidyverse)
library(evd)
library(Matrix)
library(TMB)
# Compile and load the model
TMB::compile("src/gev_single_station.cpp")
dyn.load(dynlib("src/gev_single_station"))


fn <- MakeADFun(
  data  = list(y = rgev(100, 3, 1, 0.2)),   # or your data
  parameters = list(psi = 0, tau = 0, phi = 0),
  DLL = "gev_single_station",
  silent = TRUE
)

# compare analytic vs. finite-difference gradient
library(numDeriv)
gr_fd <- grad(fn$fn, fn$par)
fn$gr()
gr_fd
max(abs(fn$gr() - gr_fd)) 



fn <- MakeADFun(
  data  = list(y = rgev(100, 3, 1, 0.2)),   # or your data
  parameters = list(psi = 0, tau = 0, phi = 0),
  DLL = "gev_single_station",
  profile = c("psi", "tau", "phi")
)

fn$fn()
fn$env$spHess()
fn$he()
