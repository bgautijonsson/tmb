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

source(here::here("R", "prepare_precip_data.R"))

# Compile and load the i.i.d. model
TMB::compile("src/gevt_multiple_stations.cpp")
dyn.load(dynlib("src/gevt_multiple_stations"))

tictoc::tic()
data <- prepare_precip_data(
  stations = read_csv(here::here("data", "stations.csv")),
  precip = read_csv(here::here("data", "precip.csv")),
  neighbors = read_csv(here::here("data", "neighbors.csv")),
  tas = read_csv(here::here("data", "tas.csv")),
  #x_range = c(0, 60),
  #y_range = c(46, 106)
)


P <- ncol(data$Y)
n_x <- length(unique(data$stations$proj_x))
n_y <- length(unique(data$stations$proj_y))

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
  gamma = rep(-2, P)
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
    u0 = 1e-1
  )
)

obj_iid$fn(obj_iid$par)
par_iid <- obj_iid$env$last.par

Q_sparse <- obj_iid$env$spHess(par_iid)
L <- Cholesky(Q_sparse, LDL = FALSE, perm = TRUE)
tictoc::toc()

max_step_results <- list(
  parameters_copula = par_iid,
  parameters_iid = par_iid,
  Hessian = Q_sparse,
  L = L
)

# Extract dimensions and parameters
stations <- data$stations
edges <- data$edges
n_stations <- nrow(data$stations)
n_param <- length(params_iid)
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

## 1.  Extract the permutation -------------------------------------------------
perm <- as.integer(L@perm) + 1 # CHOLMOD stores 0-based
# pass this straight to Stan

## 2.  Turn the factor into a *row-compressed* sparse matrix -------------------
L_C <- as(L, "sparseMatrix") # dgCMatrix (CSC)
L_R <- as(L_C, "RsparseMatrix") # dgRMatrix (CSR)  :contentReference[oaicite:0]{index=0}

## 3.  Build the CSR arrays that Stan wants ------------------------------------
row_ptr <- L_R@p + 1 # cumulative count per row (0-based)
n_values <- diff(row_ptr) # number of non-zeros per row      :contentReference[oaicite:1]{index=1}
index <- as.integer(L_R@j) + 1 # 1-based column indices for Stan
value <- as.numeric(L_R@x) # numeric data
log_det_Q <- sum(log(Matrix::diag(L)))

L_C[1:5, 1:5]
value[1:10]
index[1:10]
row_ptr[1:10] + 1
stopifnot(sum(n_values) == length(value)) # quick consistency check

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
  scaling_factor = scaling_factor
)

# Prepare initial values
psi_hat <- eta_hat[1:n_stations]
tau_hat <- eta_hat[(n_stations + 1):(2 * n_stations)]
phi_hat <- eta_hat[(2 * n_stations + 1):(3 * n_stations)]
gamma_hat <- eta_hat[(3 * n_stations + 1):(4 * n_stations)]


mu_psi <- mean(psi_hat)
mu_tau <- mean(tau_hat)
mu_phi <- mean(phi_hat)
mu_gamma <- mean(gamma_hat)


sd_psi <- sd(psi_hat)
sd_tau <- sd(tau_hat)
sd_phi <- sd(phi_hat)
sd_gamma <- sd(gamma_hat)


psi_raw <- psi_hat - mu_psi
tau_raw <- tau_hat - mu_tau
phi_raw <- phi_hat - mu_phi
gamma_raw <- gamma_hat - mu_gamma

eta_raw <- cbind(
  psi_raw / sd_psi,
  tau_raw / sd_tau,
  phi_raw / sd_phi,
  gamma_raw / sd_gamma
)

inits <- list(
  eta_raw = eta_raw,
  mu = c(mu_psi, mu_tau, mu_phi, mu_gamma),
  sigma = 4 * c(sd_psi, sd_tau, sd_phi, sd_gamma),
  rho = c(0.5, 0.5, 0.5, 0.5),
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
  iter_warmup = 300,
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

fit$summary(c("mu", "sigma", "rho")) |>
  write_csv("tables/gevt/bym2_iid.csv")

model_stations <- res$data$stations


xi_link <- function(phi) {
  c_phi <- 0.8
  # Calculate value inside the log (1 - (1/2)^c)
  inside_term <- 1 - (0.5)^c_phi

  # Calculate b_phi using the formula
  b_phi <- -c_phi^(-1) * log(inside_term) * inside_term * 2^(c_phi - 1)

  # Calculate a_phi
  a_phi <- -b_phi * log(-log(inside_term))

  xi_temp <- (phi - a_phi) / b_phi
  xi_temp <- 1.0 - exp(-exp(xi_temp))
  return(xi_temp^(1.0 / c_phi) - 0.5)
}


post_sum <- fit$summary("eta", median) |>
  rename(mcmc = median)

plot_dat <- post_sum |>
  filter(
    str_detect(variable, "^eta\\[")
  ) |>
  mutate(
    station = str_match(variable, "eta\\[(.*),.*\\]")[, 2] |> parse_number(),
    variable = str_match(variable, "eta\\[.*,(.*)\\]")[, 2] |> parse_number(),
    variable = c("psi", "tau", "phi", "gamma")[variable]
  ) |>
  mutate(
    proj_x = model_stations$proj_x,
    proj_y = model_stations$proj_y,
    .by = variable
  ) |>
  mutate(
    max_iid = res$max_results$parameters_iid
  ) |>
  pivot_longer(c(mcmc, max_iid), names_to = "type") |>
  pivot_wider(
    names_from = variable
  ) |>
  mutate(
    mu = exp(psi),
    sigma = exp(tau + psi),
    xi = xi_link(phi),
    Delta = exp(gamma)
  ) |>
  pivot_longer(
    c(psi, tau, phi, mu, sigma, xi, gamma, Delta),
    names_to = "variable"
  ) |>
  mutate(
    type = fct_relevel(type, "max_iid", "mcmc") |>
      fct_recode(
        "Smooth (IID)" = "mcmc",
        "Max (IID)" = "max_iid"
      ),
    variable = fct_relevel(
      variable,
      "psi",
      "tau",
      "phi",
      "gamma",
      "mu",
      "sigma",
      "xi",
      "Delta"
    )
  )

plot_dat |>
  write_csv("tables/gevt/plot_dat_iid.csv")

plot_dat <- read_csv("tables/gevt/plot_dat_iid.csv")

plot_dat |>
  filter(
    # str_detect(type, "Max")
  ) |>
  mutate(
    variable = fct_relevel(
      variable,
      "psi",
      "tau",
      "phi",
      "gamma",
      "mu",
      "sigma",
      "xi",
      "Delta"
    )
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
  facet_grid(
    rows = vars(type),
    cols = vars(variable),
    scales = "free",
    labeller = label_parsed
  ) +
  theme(
    legend.position = "none",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = "Comparing Smooth-Step Results to Max-Step Results"
  )

ggsave(
  filename = "Figures/gevt/iid/comparison.png",
  width = 8,
  height = 0.4 * 8,
  scale = 1.5
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
          scale_fill_distiller(
            palette = "RdBu"
          ) +
          scale_colour_distiller(
            palette = "RdBu"
          ) +
          labs(
            subtitle = unique(x$model)
          )
      }
    ) |>
    wrap_plots(ncol = 1) +
    plot_annotation(
      # title = "Spatial distribution of GEV parameters",
      subtitle = latex2exp::TeX(
        str_c(
          "$",
          var,
          "$"
        )
      )
    )

  ggsave(
    plot = p,
    filename = str_c("Figures/gevt/iid/", var, ".png"),
    width = 8,
    height = 2 * 8,
    scale = 0.6
  )
}

c("psi", "mu", "tau", "sigma", "phi", "xi", "gamma", "Delta") |>
  map(make_plot)


plot_dat |>
  mutate(
    variable = fct_relevel(
      variable,
      "psi",
      "tau",
      "phi",
      "gamma",
      "mu",
      "sigma",
      "xi",
      "Delta"
    )
  ) |>
  pivot_wider(names_from = type) |>
  janitor::clean_names() |>
  mutate(
    variable2 = variable
  ) |>
  group_by(variable2) |>
  group_map(
    \(x, ...) {
      lower <- min(c(x$max_iid, x$smooth_iid))
      upper <- max(c(x$max_iid, x$smooth_iid))

      x |>
        ggplot(aes(max_iid, smooth_iid)) +
        geom_abline(
          intercept = 0,
          slope = 1,
          lty = 2
        ) +
        geom_point(
          alpha = 0.1
        ) +
        scale_x_continuous(
          guide = ggh4x::guide_axis_truncated(),
          breaks = scales::breaks_extended(7)
        ) +
        scale_y_continuous(
          guide = ggh4x::guide_axis_truncated(),
          breaks = scales::breaks_extended(7)
        ) +
        coord_cartesian(
          xlim = c(lower, upper),
          ylim = c(lower, upper)
        ) +
        labs(
          subtitle = latex2exp::TeX(
            str_c(
              "$",
              unique(x$variable),
              "$"
            )
          ),
          x = "Max-Step",
          y = "Smooth-Step"
        )
    }
  ) |>
  wrap_plots(nrow = 2) +
  plot_annotation(
    title = "Comparing station-wise estimates from the Max and Smooth-steps",
    subtitle = str_c(
      "Location, scale and shape on the unconstrained (upper row) and constrained (lower row) scale"
    )
  )

ggsave(
  filename = "Figures/gevt/iid/max_smooth_compare.png",
  width = 8,
  height = 0.621 * 8,
  scale = 1.3
)
