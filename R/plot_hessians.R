# R code
library(tidyverse)
library(evd)
library(Matrix)
library(TMB)
library(Rcpp)
library(RcppEigen)
library(bayesplot)
library(posterior)
library(patchwork)
theme_set(
  bggjphd::theme_bggj()
)

source(here::here("R", "prepare_precip_data.R"))
source(here::here("R", "fit_copula.R"))

# Compile and load the copula model
TMB::compile("src/gevt_multiple_stations_with_copula_precision_matrix.cpp")
dyn.load(dynlib("src/gevt_multiple_stations_with_copula_precision_matrix"))

# Compile and load the i.i.d. model
TMB::compile("src/gevt_multiple_stations.cpp")
dyn.load(dynlib("src/gevt_multiple_stations"))

stations <- read_csv(here::here("data", "stations.csv"))
precip <- read_csv(here::here("data", "precip.csv"))
neighbors <- read_csv(here::here("data", "neighbors.csv"))
tas <- read_csv(here::here("data", "tas.csv"))

data <- prepare_precip_data(
  stations = stations,
  precip = precip,
  neighbors = neighbors,
  tas = tas,
  x_range = c(0, 6),
  y_range = c(46, 52)
)
P <- ncol(data$Y)
n_x <- length(unique(data$stations$proj_x))
n_y <- length(unique(data$stations$proj_y))

nu <- 0
rho <- fit_copula(data$Y, n_x, n_y, nu = nu)
rho <- c(0.5, 0.5)

Q <- stdmatern::make_standardized_matern_eigen(n_x, n_y, rho[1], rho[2], nu)

precision_components <- prepare_precision(
  rho,
  n_x,
  n_y,
  nu
)

prior_mu <- c(2.5, -1, 0, -2.5)
prior_sigma <- c(2, 2, 0.2, 1)

data_list_iid <- list(
  y = data$Y,
  X = data$X,
  prior_mu = prior_mu,
  prior_sigma = prior_sigma
)

params_iid <- list(
  psi = rep(2, P),
  tau = rep(-1, P),
  phi = rep(0.05, P),
  gamma = rep(0, P)
)

# Create TMB object for i.i.d. model
obj_iid <- MakeADFun(
  data = data_list_iid,
  parameters = params_iid,
  DLL = "gevt_multiple_stations",
  hessian = TRUE,
  profile = c("phi", "tau", "psi", "gamma"),
  inner.control = list(
    maxit = 1e4,
    tol = 1e-10,
    silent = FALSE,
    tol10 = 1e-4,
    power = 0.9,
    u0 = 1e-2
  )
)

obj_iid$fn(obj_iid$par)
par_iid <- obj_iid$env$last.par
Q_iid <- obj_iid$env$spHess(par_iid)

# Prepare data and parameters
data_list_copula <- list(
  y = data$Y,
  X = data$X,
  Q = Q,
  prior_mu = prior_mu,
  prior_sigma = prior_sigma
)


params_copula <- list(
  psi = par_iid[1:P],
  tau = par_iid[(P + 1):(2 * P)],
  phi = par_iid[(2 * P + 1):(3 * P)],
  gamma = par_iid[(3 * P + 1):(4 * P)]
)


# Create TMB object for copula model
obj_copula <- MakeADFun(
  data = data_list_copula,
  parameters = params_copula,
  DLL = "gevt_multiple_stations_with_copula_precision_matrix",
  # silent = TRUE,
  hessian = TRUE,
  profile = c("phi", "tau", "psi", "gamma"),
  inner.control = list(
    maxit = 1e3,
    tol = 1e-10,
    silent = FALSE,
    tol10 = 1e-4,
    power = 0.2,
    u0 = 1e-2
  )
)


obj_copula$fn(obj_copula$par)
par_copula <- obj_copula$env$last.par.best
Q_copula <- obj_copula$env$spHess(par_copula)


image(Q_iid)
image(Q_copula)

L_iid <- Cholesky(Q_iid, LDL = FALSE, perm = TRUE)
L_copula <- Cholesky(Q_copula, LDL = FALSE, perm = TRUE)
image(L_iid)
image(L_copula)
