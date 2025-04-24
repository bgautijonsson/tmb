# R code
library(tidyverse)
library(evd)
library(Matrix)
library(TMB)
library(Rcpp)
library(RcppEigen)
library(maxandsmooth)
library(bayesplot)
library(posterior)
library(patchwork)
theme_set(
  bggjphd::theme_bggj()
)



# Compile and load the copula model
TMB::compile("src/gev_multiple_stations_with_copula_precision_matrix.cpp")
dyn.load(dynlib("src/gev_multiple_stations_with_copula_precision_matrix"))

# Compile and load the i.i.d. model
TMB::compile("src/gev_multiple_stations.cpp")
dyn.load(dynlib("src/gev_multiple_stations"))

tictoc::tic()
data <- prepare_precip_data(
  stations = bggjphd::stations,
  precip = bggjphd::precip,
  # x_range = c(0, 20),
  # y_range = c(46, 66)
)
P <- ncol(data$Y)

n_x <- length(unique(data$stations$proj_x))
n_y <- length(unique(data$stations$proj_y))
nu <- 0

rho <- fit_copula(data$Y, n_x, n_y, nu)
Q <- stdmatern::make_standardized_matern_eigen(n_x, n_y, rho[1], rho[2], nu)

precision_components <- prepare_precision(
  rho,
  n_x,
  n_y,
  nu
)

data_list_iid <- list(
  y = data$Y,
  prior_mu = c(0, 0, 0),
  prior_sigma = c(sqrt(10), sqrt(4), sqrt(2))
)

params_iid <- list(
  psi = rep(0.7, P),
  tau = rep(0.2, P),
  phi = rep(0.1, P)
)

# Create TMB object for i.i.d. model
obj_iid <- MakeADFun(
  data = data_list_iid,
  parameters = params_iid,
  DLL = "gev_multiple_stations",
  hessian = TRUE,
  profile = c("phi", "tau", "psi"),
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

mean_mu <- mean(par_iid[1:P])
mean_tau <- mean(par_iid[(P + 1):(2 * P)])
mean_xi <- mean(par_iid[(2 * P + 1):(3 * P)])

sd_mu <- sd(par_iid[1:P])
sd_tau <- sd(par_iid[(P + 1):(2 * P)])
sd_xi <- sd(par_iid[(2 * P + 1):(3 * P)])

# Prepare data and parameters
data_list_copula <- list(
  y = data$Y,
  Q = Q,
  prior_mu = c(mean_mu, mean_tau, mean_xi),
  prior_sigma = 4 * c(sd_mu, sd_tau, sd_xi)
)


params_copula <- list(
  psi = par_iid[1:P],
  tau = par_iid[(P + 1):(2 * P)],
  phi = par_iid[(2 * P + 1):(3 * P)]
)


# Create TMB object for copula model
obj_copula <- MakeADFun(
  data = data_list_copula,
  parameters = params_copula,
  DLL = "gev_multiple_stations_with_copula_precision_matrix",
  # silent = TRUE,
  hessian = TRUE,
  profile = c("phi", "tau", "psi"),
  inner.control = list(
    maxit = 1e3,
    tol = 1e-10,
    silent = FALSE,
    tol10 = 1e-4,
    power = 0.2,
    u0 = 1e-2
  )
)

# Add a custom optimizer wrapper with gradient scaling
# custom_optim <- function(par, fn, gr, ...) {
#   scaled_gr <- function(x) {
#     g <- gr(x)
#     # Scale down extremely large gradient components
#     max_grad <- max(abs(g))
#     if (max_grad > 1000) {
#       scale_factor <- 1000 / max_grad
#       g <- g * scale_factor
#     }
#     return(g)
#   }
#   nlminb(par, fn, scaled_gr, ...)
# }
# 
# # Use the custom optimizer
# opt <- custom_optim(obj_copula$par, obj_copula$fn, obj_copula$gr, 
#                     control = list(eval.max = 1000, iter.max = 500))
# 
# as.numeric(obj_copula$gr(obj_copula$env$last.par))
# obj_copula$env$last.par[!is.finite(as.numeric(obj_copula$gr(obj_copula$env$last.par)))]
# x <- obj_copula$env$last.par
# y <- as.numeric(obj_copula$gr(obj_copula$env$last.par))
# which(!is.finite(y))
# y[!is.finite(y)] <- 1e4
# plot(x, y)
# compare analytic vs. finite-difference gradient
# library(numDeriv)
# gr_fd <- grad(obj_copula$fn, obj_copula$par)
# obj_copula$gr()
# gr_fd
# max(abs(obj_copula$gr() - gr_fd)) 

obj_copula$fn(obj_copula$par)
par_copula <- obj_copula$env$last.par.best
par_copula[which.min(par_copula)]
par_copula[which.max(par_copula)]
Q_sparse <- obj_copula$env$spHess(par_copula)
L <- Cholesky(Q_sparse, LDL = FALSE, perm = TRUE)
tictoc::toc()



max_step_results <- list(
  parameters_copula = par_copula,
  parameters_iid = par_iid,
  Hessian = Q_sparse,
  L = L,
  rho = rho
)

# Extract dimensions and parameters
stations <- data$stations
edges <- data$edges
n_stations <- nrow(data$stations)
n_param <- 3
eta_hat <- max_step_results$parameters_copula
# L <- max_step_results$L


# Process edges
n_edges <- nrow(edges)
node1 <- edges$station
node2 <- edges$neighbor

# Get grid dimensions
dim1 <- length(unique(stations$proj_x))
dim2 <- length(unique(stations$proj_y))



# Process Cholesky components
# value <- L@x
# index <- L@i + 1
# n_values <- L@nz
# log_det_Q <- sum(log(Matrix::diag(L)))
# perm <- L@perm + 1

## 2.  Extract the permutation -------------------------------------------------
perm <- as.integer(L@perm) + 1               # CHOLMOD stores 0-based
# pass this straight to Stan

## 3.  Turn the factor into a *row-compressed* sparse matrix -------------------
L_C  <- as(L, "sparseMatrix")                 # dgCMatrix (CSC)
L_R  <- as(L_C, "RsparseMatrix")                # dgRMatrix (CSR)  :contentReference[oaicite:0]{index=0}

## 4.  Build the CSR arrays that Stan wants ------------------------------------
row_ptr  <- L_R@p + 1                           # cumulative count per row (0-based)
n_values <- diff(row_ptr)                       # number of non-zeros per row      :contentReference[oaicite:1]{index=1}
index    <- as.integer(L_R@j) + 1               # 1-based column indices for Stan
value    <- as.numeric(L_R@x)                   # numeric data
log_det_Q <- sum(log(Matrix::diag(L)))
L_C[1:5, 1:5]
value[1:10]
index[1:10]
row_ptr[1:10] + 1
stopifnot(sum(n_values) == length(value))       # quick consistency check
# Calculate scaling factor for BYM2 model
scaling_factor <- get_scaling_factor(edges, n_stations)


# Prepare Stan data
stan_data <- list(
  n_stations = n_stations,
  n_param = n_param,
  n_obs = 2,
  eta_hat = eta_hat,
  perm = perm,
  n_edges = n_edges,
  node1 = node1,
  node2 = node2,
  n_nonzero_chol_Q = sum(n_values),
  n_values = n_values,
  index = index,
  value = value,
  row_ptr = row_ptr,
  log_det_Q = log_det_Q,
  dim1 = dim1,
  dim2 = dim2,
  nu = nu,
  scaling_factor = scaling_factor
)

# Prepare initial values
psi_hat <- eta_hat[1:n_stations]
tau_hat <- eta_hat[(n_stations + 1):(2 * n_stations)]
phi_hat <- eta_hat[(2 * n_stations + 1):(3 * n_stations)]

mu_psi <- mean(psi_hat)
mu_tau <- mean(tau_hat)
mu_phi <- mean(phi_hat)

sd_psi <- sd(psi_hat)
sd_tau <- sd(tau_hat)
sd_phi <- sd(phi_hat)

psi_raw <- psi_hat - mu_psi
tau_raw <- tau_hat - mu_tau
phi_raw <- phi_hat - mu_phi

eta_raw <- cbind(psi_raw / sd_psi, tau_raw / sd_tau, phi_raw / sd_phi)

inits <- list(
  psi = psi_hat,
  tau = tau_hat,
  phi = phi_hat,
  mu_psi = mu_psi,
  mu_tau = mu_tau,
  mu_phi = mu_phi,
  eta_raw = eta_raw,
  mu = c(mu_psi, mu_tau, mu_phi),
  sigma = c(1, 1, 1),
  rho = c(0.5, 0.5, 0.5),
  eta_spatial = eta_raw,
  eta_random = matrix(0, nrow = nrow(eta_raw), ncol = ncol(eta_raw))
)

# Compile and run Stan model
model <- cmdstanr::cmdstan_model(
  here::here("stan", "stan_smooth_bym2.stan")
)

fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  refresh = 100,
  iter_warmup = 1000,
  iter_sampling = 1000,
  init = rep(list(inits), 4)
)


res <- list(
  data = data,
  max_results = max_step_results,
  smooth_results = fit
)

# res$smooth_results$draws()
# write_rds(res, "results.rds")

fit <- res$smooth_results

fit$summary(c("mu", "sigma", "rho"))

model_stations <- res$data$stations




post_sum <- fit$summary("eta", median) |>
  rename(mcmc = median)

plot_dat <- post_sum |>
  filter(
    str_detect(variable, "^eta\\[")
  ) |>
  mutate(
    station = str_match(variable, "eta\\[(.*),.*\\]")[, 2] |> parse_number(),
    variable = str_match(variable, "eta\\[.*,(.*)\\]")[, 2] |> parse_number(),
    variable = c("psi", "tau", "phi")[variable]
  ) |>
  mutate(
    proj_x = model_stations$proj_x,
    proj_y = model_stations$proj_y,
    .by = variable
  ) |>
  mutate(
    max_iid = res$max_results$parameters_iid,
    max_copula = res$max_results$parameters_copula
  ) |>
  pivot_longer(c(mcmc, max_iid, max_copula), names_to = "type") |>
  pivot_wider(
    names_from = variable
  ) |>
  mutate(
    mu = exp(psi),
    sigma = exp(tau + psi),
    xi = plogis(phi)
  ) |>
  pivot_longer(c(psi, tau, phi, mu, sigma, xi), names_to = "variable") |>
  mutate(
    type = fct_relevel(type, "max_iid", "max_copula", "mcmc") |>
      fct_recode(
        "Smooth (MCMC)" = "mcmc",
        "IID (Max)" = "max_iid",
        "Copula (Max)" = "max_copula"
      ),
    variable = fct_relevel(
      variable,
      "psi", "tau", "phi",
      "mu", "sigma", "xi"
    )
  )

plot_dat |>
  write_csv("plot_dat.csv")

plot_dat |>
  filter(
    # str_detect(type, "Max")
  ) |>
  ggplot(aes(value, after_stat(scaled))) +
  geom_density(
    data = ~ rename(.x, tp = type),
    aes(group = tp),
    alpha = 0.2,
    fill = "grey60"
  ) +
  geom_density(
    aes(fill = type),
    alpha = 0.7
  ) +
  scale_x_continuous(
    expand = expansion(mult = 0.2)
  ) +
  scale_fill_brewer(
    palette = "Set1"
  ) +
  facet_wrap(
    vars(variable, type),
    scales = "free",
    ncol = 3
  ) +
  theme(
    legend.position = "none",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank()
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = "Comparing Smooth-Step Results to Max-Step Results With/Without Copula"
  )

ggsave(
  filename = "comparison.png",
  width = 8,
  height = 1.2 * 8,
  scale = 1.2
)

uk <- bggjphd::get_uk_spatial(scale = "large")

d <- bggjphd::stations |>
  bggjphd::stations_to_sf() |>
  bggjphd::points_to_grid() |>
  inner_join(
    plot_dat,
    by = join_by(proj_x, proj_y)
  ) |>
  select(-contains("station"))

make_plot <- function(var) {
  p <- d |>
    filter(
      variable == var
    ) |>
    rename(
      model = type
    ) |> 
    mutate(
      model2 = model
    ) |> 
    group_by(model2) |> 
    group_map(
      \(x, ...) {
        x |> 
          ggplot() +
          geom_sf(
            data = uk |> filter(name == "Ireland")
          ) +
          geom_sf(
            aes(fill = value, col = value),
            linewidth = 0.01,
            alpha = 0.6
          ) +
          scale_fill_viridis_c() +
          scale_colour_viridis_c() +
          labs(
            subtitle = unique(x$model)
          )
      }
    ) |> 
    wrap_plots() +
    plot_annotation(
      title = "Spatial distribution",
      subtitle = var
    )
  
  ggsave(
    plot = p,
    filename = str_c(var, ".png"),
    width = 8,
    height = 0.4 * 8,
    scale = 1.8
  )
}

c("psi", "mu", "tau", "sigma", "phi", "xi") |>
  map(make_plot)
