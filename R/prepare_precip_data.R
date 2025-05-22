#' Update station names in a data frame
#'
#' @param table Data frame to update
#' @param variable Variable containing station names
#' @param new_names Data frame with station to new_name mapping
#' @return Updated data frame
#' @importFrom dplyr inner_join mutate select
#' @keywords internal
update_names <- function(table, variable, new_names) {
  table |>
    dplyr::inner_join(
      new_names,
      by = dplyr::join_by({{ variable }} == station)
    ) |>
    dplyr::mutate(
      "{{variable}}" := new_name
    ) |>
    dplyr::select(-new_name)
}

#' Prepare precipitation data for Max-and-Smooth analysis
#'
#' @param stations Data frame containing station information
#' @param precip Data frame containing precipitation data
#' @param x_range Numeric vector of length 2 for x-axis range
#' @param y_range Numeric vector of length 2 for y-axis range
#'
#' @return A list containing:
#'   \item{Y}{Matrix of precipitation data}
#'   \item{stations}{Filtered stations data frame}
#'   \item{new_names}{Station name mapping}
#'   \item{edges}{Edge list for spatial neighborhood}
#'
#' @importFrom dplyr filter between semi_join mutate distinct select inner_join join_by
#' @importFrom tidyr pivot_wider
#' @export
prepare_precip_data <- function(
  stations,
  precip,
  neighbors,
  tas,
  x_range = c(0, 70),
  y_range = c(46, 136)
) {
  # Filter stations by location
  model_stations <- stations |>
    dplyr::filter(
      dplyr::between(proj_x, x_range[1], x_range[2]),
      dplyr::between(proj_y, y_range[1], y_range[2])
    )

  # Create new station names
  new_names <- model_stations |>
    dplyr::mutate(new_name = dplyr::row_number()) |>
    dplyr::distinct(station, new_name)

  # Filter and reshape precipitation data
  model_precip <- precip |>
    dplyr::semi_join(model_stations, by = "station")

  model_tas <- tas |>
    dplyr::semi_join(model_stations, by = "station")

  Y <- model_precip |>
    tidyr::pivot_wider(names_from = station, values_from = precip) |>
    dplyr::select(-year) |>
    as.matrix()

  X <- model_tas |>
    dplyr::mutate(
      tas = tas - tas[year == min(year)],
      .by = station
    ) |>
    tidyr::pivot_wider(names_from = station, values_from = tas) |>
    dplyr::select(-year) |>
    as.matrix()

  # Prepare edges with updated names
  edges <- neighbors |>
    dplyr::filter(
      type %in% c("e", "n", "w", "s")
    ) |>
    dplyr::inner_join(
      model_stations,
      by = dplyr::join_by(station)
    ) |>
    dplyr::semi_join(
      model_stations,
      by = dplyr::join_by(neighbor == station)
    ) |>
    dplyr::select(station, neighbor) |>
    update_names(station, new_names) |>
    update_names(neighbor, new_names)

  # Return prepared data
  list(
    Y = Y,
    X = X,
    stations = model_stations,
    new_names = new_names,
    edges = edges
  )
}


#' Helper function to calculate scaling factor for BYM2 model
#'
#' @param edges Data frame containing neighborhood structure
#' @param N Number of stations
#' @return Scaling factor for BYM2 model
#' @importFrom Matrix sparseMatrix Diagonal rowSums diag
#' @importFrom INLA inla.qinv
get_scaling_factor <- function(edges, N) {
  # Filter and rename edges
  nbs <- edges |>
    dplyr::filter(neighbor > station) |>
    dplyr::rename(node1 = station, node2 = neighbor)

  # Create adjacency matrix
  adj.matrix <- Matrix::sparseMatrix(
    i = nbs$node1,
    j = nbs$node2,
    x = 1,
    symmetric = TRUE
  )

  # Create ICAR precision matrix
  Q <- Matrix::Diagonal(N, Matrix::rowSums(adj.matrix)) - adj.matrix

  # Add small jitter for numerical stability
  Q_pert <- Q +
    Matrix::Diagonal(N) * max(Matrix::diag(Q)) * sqrt(.Machine$double.eps)

  # Compute inverse with sum-to-zero constraint
  Q_inv <- INLA::inla.qinv(
    Q_pert,
    constr = list(A = matrix(1, 1, N), e = 0)
  )

  # Return geometric mean of variances
  exp(mean(log(Matrix::diag(Q_inv))))
}

#' Prepare precision matrix components for GEV fitting
#'
#' @param parameters A vector of length 2 containing optimized parameters (rho1, rho2)
#' @param n_x Number of x-axis grid points
#' @param n_y Number of y-axis grid points
#' @param nu Smoothness parameter for the Matern covariance (default = 2)
#'
#' @return A list containing:
#'   \item{index}{Row indices of non-zero elements in L}
#'   \item{n_values}{Number of non-zero elements per column in L}
#'   \item{values}{Non-zero values in L}
#'   \item{log_det}{Log determinant of L}
#'
#' @export
prepare_precision <- function(rho, n_x, n_y, nu = 2) {
  # Extract parameters
  rho1 <- rho[1]
  rho2 <- rho[2]

  # Compute precision matrix and its Cholesky decomposition
  Q <- stdmatern::make_standardized_matern_eigen(n_x, n_y, rho1, rho2, nu)
  L <- Matrix::t(Matrix::chol(Q))

  # Extract sparse matrix information
  n_values <- Matrix::colSums(L != 0)
  index <- attributes(L)$i
  values <- attributes(L)$x
  log_det <- sum(log(Matrix::diag(L)))

  # Return components needed for GEV fitting
  list(
    index = index,
    n_values = n_values,
    values = values,
    log_det = log_det
  )
}
