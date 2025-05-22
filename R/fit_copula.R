#' Fit a Matern copula to precipitation data
#'
#' @param data A matrix of precipitation data where rows are observations and columns are locations
#' @param n_x Number of x-axis grid points
#' @param n_y Number of y-axis grid points
#' @param nu Smoothness parameter for the Matern covariance (default = 2)
#' @param init Initial values for optimization (default = c(0.2, 0.4))
#'
#' @return A list containing:
#'   \item{parameters}{Optimized parameters (rho1, rho2)}
#'
#' @importFrom stats optim
#' @export
fit_copula <- function(
  Y,
  n_x,
  n_y,
  nu = 2,
  init = qlogis(c(0.5, 0.5))
) {
  # Convert data to quantiles
  Y_quantiles <- apply(Y, 2, function(x) rank(x) / (length(x) + 1)) |>
    qnorm()

  # Optimize the Matern copula parameters
  par <- optim(
    init,
    fn = function(par) {
      rho1 <- plogis(par[1])
      rho2 <- plogis(par[2])
      -sum(stdmatern::dmatern_copula_eigen(
        Y_quantiles,
        n_x,
        n_y,
        rho1,
        rho2,
        nu
      ))
    }
  )$par

  par <- plogis(par)

  # Return the optimized parameters
  return(par)
}

#' Fit a Matern t-copula to precipitation data
#'
#' @param data A matrix of precipitation data where rows are observations and columns are locations
#' @param n_x Number of x-axis grid points
#' @param n_y Number of y-axis grid points
#' @param nu Smoothness parameter for the Matern covariance (default = 2)
#' @param init Initial values for optimization (default = c(0.2, 0.4))
#' @param df Degrees of freedom for the t-distribution (default = 4)
#'
#' @return A list containing:
#'   \item{parameters}{Optimized parameters (rho1, rho2)}
#'
#' @importFrom stats optim
#' @export
fit_tcopula <- function(
  Y,
  n_x,
  n_y,
  nu = 2,
  init = qlogis(c(0.1, 0.1)),
  df = 4
) {
  # Convert data to quantiles
  Y_quantiles <- apply(Y, 2, function(x) rank(x) / (length(x) + 1)) |>
    qt(df = df)

  # Optimize the Matern copula parameters
  par <- optim(
    init,
    fn = function(par) {
      rho1 <- plogis(par[1])
      rho2 <- plogis(par[2])
      -sum(stdmatern::dmatern_tcopula_eigen(
        Y_quantiles,
        n_x,
        n_y,
        rho1,
        rho2,
        nu,
        df
      ))
    }
  )$par

  par <- plogis(par)

  # Return the optimized parameters
  return(par)
}
