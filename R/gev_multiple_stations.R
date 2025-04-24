# R code
library(tidyverse)
library(evd)
library(Matrix)
library(TMB)


# Compile and load the model
TMB::compile("src/gev_multiple_stations.cpp")
dyn.load(dynlib("src/gev_multiple_stations"))


set.seed(123)
N <- 60
P <- 100

# True parameters for each station (could vary by station)
mu_true <- rep(3, P)
sigma_true <- rep(1, P)
xi_true <- rep(0.2, P)
data <- matrix(NA, nrow = N, ncol = P)
for (p in 1:P) {
  data[, p] <- rgev(N, mu_true[p], sigma_true[p], xi_true[p])
}


# Prepare data and parameters
data_list <- list(
  y = data,
  prior_mu = c(0, 0, 0),
  prior_sigma = c(sqrt(10), sqrt(4), sqrt(2))
)
params <- list(
  psi = rep(0.7, P),
  tau = rep(0.2, P),
  phi = rep(0.1, P)
)

openmp(n = 4, DLL = "gev_multiple_stations")
# Create TMB object
obj <- MakeADFun(
  data = data_list,
  parameters = params,
  DLL = "gev_multiple_stations",
  # silent = TRUE,
  hessian = TRUE,
  profile = c("phi", "tau", "psi"),
  # inner.control = list(
  #   maxit = 1e4,
  #   tol = 1e-10,
  #   silent = FALSE,
  #   tol10 = 1e-4,
  #   power = 0.9,
  #   u0 = 1e-3
  # )
)

obj$fn(obj$par)
par <- obj$env$last.par
par[which.min(par)]
par[which.max(par)]
plot(par)
Q_sparse <- obj$env$spHess(par)
Q_sparse |> image()
L <- chol(Q_sparse)
L |> image()


