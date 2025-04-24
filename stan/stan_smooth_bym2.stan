functions {

  /*
  * Sparse matrix-vector multiplication using compressed sparse row (CSR) format
  *
  * This function uses Stan's built-in CSR matrix operations to efficiently multiply
  * a sparse matrix by a vector. The CSR format stores only non-zero elements along 
  * with their indices, making it memory-efficient for sparse matrices commonly 
  * encountered in spatial statistics and network models.
  *
  * @param rows The number of rows in the matrix
  * @param w Vector of non-zero values in the matrix (stored in row-major order)
  * @param v Array of column indices for each non-zero element
  * @param u Array of indices indicating where each row starts in the w and v arrays
  * @param x Vector to multiply with the sparse matrix
  * @return Vector result of multiplying the sparse matrix by x
  */
  vector L_times(
    int rows, 
    vector w, 
    array[] int v, 
    array[] int u,
    vector x
  ) {
    return csr_matrix_times_vector(rows, rows, w, v, u, x);
  }

  /*
  * Intrinsic Conditional Autoregressive (ICAR) prior density function
  * 
  * This implements a spatial smoothing prior for areal data, commonly used in disease mapping.
  * The ICAR component ensures spatial autocorrelation by penalizing differences between 
  * neighboring regions. The normal term for the sum ensures identifiability.
  *
  * @param phi Vector of spatial random effects
  * @param N Number of edges in the spatial adjacency graph
  * @param node1 Array of origin nodes for each edge
  * @param node2 Array of destination nodes for each edge
  * @return Log probability density
  */
  real icar_normal_lpdf(vector phi, int N, array[] int node1, array[] int node2) {
    // Penalize differences between connected regions
    return - 0.5 * dot_self((phi[node1] - phi[node2])) +
      // Soft sum-to-zero constraint for identifiability
      normal_lpdf(sum(phi) | 0, 0.001 * N);
  }

  /*
  * Multivariate normal log probability density using sparse Cholesky factor of precision matrix
  *
  * This function efficiently computes the log probability of a multivariate normal distribution
  * with a sparse precision matrix. It uses the Cholesky decomposition for computational stability
  * and takes advantage of sparsity patterns to improve performance.
  *
  * @param y Vector of observed values
  * @param x Vector of means
  * @param n_values Array containing number of non-zero elements in each row of Cholesky factor
  * @param index Array of column indices for non-zero elements
  * @param values Vector of non-zero values in the Cholesky factor
  * @param log_det Log determinant of the precision matrix
  * @param perm Permutation vector from the Cholesky decomposition
  * @return Log probability density
  */
  real normal_prec_chol_lpdf(
    vector y, vector x, 
    array[] int row_ptr, array[] int index, 
    vector values, real log_det, array[] int perm
  ) {
    int N = num_elements(x);
    vector[N] q = rep_vector(0, N);
    // Permute the difference between observed and expected values
    vector[N] diff = y[perm] - x[perm];
    
    q = L_times(N, values, index, row_ptr, diff);

    // Log density = log|Q|/2 - (1/2)(y-x)'Q(y-x) = log|L| - q'q/2
    return log_det - dot_self(q) / 2;
  }
}

data {
  // Basic dimensions
  int<lower = 1> n_stations;  // Number of spatial locations/regions
  int<lower = 1> n_obs;       // Number of observations
  int<lower = 1> n_param;     // Number of parameters per location (typically 3 for GEV model)

  // Parameter estimates from previous model fitting
  vector[n_stations * n_param] eta_hat;  // Parameter estimates stacked by parameter type
  
  // Spatial adjacency information
  int<lower = 1> n_edges;  // Number of edges in adjacency graph
  array[n_edges] int<lower = 1, upper = n_stations> node1;  // Origin nodes
  array[n_edges] int<lower = 1, upper = n_stations> node2;  // Destination nodes
  real<lower = 0> scaling_factor;  // Scaling factor for BYM2 spatial component
  
  // Sparse Cholesky factor information for precision matrix
  int<lower = 1> n_nonzero_chol_Q;  // Number of non-zero elements in Cholesky factor
  array[n_param * n_stations] int n_values;  // Number of non-zero elements per row
  array[n_nonzero_chol_Q] int index;  // Column indices of non-zero elements
  vector[n_nonzero_chol_Q] value;  // Values of non-zero elements
  array[n_stations * n_param] int<lower=1> perm;  // Permutation indices for Cholesky decomposition
  array[n_param * n_stations + 1] int row_ptr;  // Row pointers for CSR format
  real<lower = 0> log_det_Q;  // Log determinant of precision matrix


}

transformed data {
  // Extract parameter estimates by type for easier handling
  vector[n_stations] psi_hat = eta_hat[1:n_stations];  // Location parameter estimates
  vector[n_stations] tau_hat = eta_hat[(n_stations + 1):(2 * n_stations)];  // Scale parameter estimates
  vector[n_stations] phi_hat = eta_hat[(2 * n_stations + 1):(3 * n_stations)];  // Shape parameter estimates  
}

parameters {
  // BYM2 model components
  matrix[n_stations, n_param] eta_spatial;  // Spatial random effects (structured)
  matrix[n_stations, n_param] eta_random;   // Independent random effects (unstructured)
  vector[n_param] mu;         // Overall mean for each parameter type
  vector<lower = 0>[n_param] sigma;  // Overall standard deviation for each parameter type
  vector<lower = 0, upper = 1>[n_param] rho;  // Mixing proportion between spatial and random components
}  

model {
  vector[n_param * n_stations] eta;  // Combined spatial effects for all parameters

  // Build the linear predictor for each parameter type
  for (p in 1:n_param) {
    int start = ((p - 1) * n_stations + 1);
    int end = (p * n_stations);
    
    // BYM2 parameterization: combines spatial and unstructured random effects
    // with proper scaling for interpretability
    eta[start:end] = mu[p] + sigma[p] * 
      (sqrt(rho[p] / scaling_factor) * eta_spatial[, p] + 
       sqrt(1 - rho[p]) * eta_random[, p]);

    // Add spatial smoothing prior for structured component
    target += icar_normal_lpdf(eta_spatial[, p] | n_edges, node1, node2);
  }

  // Priors for BYM2 model components
  target += std_normal_lpdf(to_vector(eta_random));  // Independent random effects
  target += exponential_lpdf(sigma | 1);             // Overall standard deviations
  target += beta_lpdf(rho | 1, 1);                   // Mixing proportions (uniform prior)

  // Likelihood using sparse precision matrix
  target += normal_prec_chol_lpdf(
    eta_hat | eta, row_ptr, index, value, log_det_Q, perm
  );
}

generated quantities {
  // Reconstruct the combined spatial effects for reporting/visualization
  matrix[n_stations, n_param] eta;
  for (p in 1:n_param) {
    int start = ((p - 1) * n_stations + 1);
    int end = (p * n_stations);
    
    // Same formula as in the model block for consistency
    eta[, p] = mu[p] + sigma[p] * (sqrt(rho[p] / scaling_factor) * eta_spatial[, p] + sqrt(1 - rho[p]) * eta_random[, p]);
  }
}
