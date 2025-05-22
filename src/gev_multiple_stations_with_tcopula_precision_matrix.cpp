#include <TMB.hpp>

// Link function for location parameter (mu)
/**
 * @brief Link function for GEV location parameter
 * 
 * Transforms the unconstrained parameter ψ to the location parameter μ
 * using an exponential transformation: μ = exp(ψ)
 * 
 * @param psi Unconstrained parameter
 * @return Type Transformed location parameter
 */
template<class Type>
Type mu_link(Type psi) {
    return exp(psi);
}

// Link function for scale parameter (sigma)
/**
 * @brief Link function for GEV scale parameter
 * 
 * Transforms unconstrained parameters (ψ,τ) to positive scale parameter σ
 * using: σ = exp(ψ + τ)
 * 
 * @param psi Location parameter contribution
 * @param tau Scale-specific parameter
 * @return Type Transformed scale parameter (always positive)
 */
template<class Type>
Type sigma_link(Type psi,Type tau) {
    return exp(psi + tau);
}

// Link function for shape parameter (xi)
template<class Type>
Type xi_link(Type phi, Type a_phi, Type b_phi, Type c_phi) {
    // Calculate xi = (1 - exp(-exp((phi - a_phi) / b_phi))) ^ (1/c_phi) - 0.5
    Type z = (phi - a_phi) / b_phi;
    Type u = Type(1.0) - exp(-exp(z));
    Type xi = pow(u, Type(1.0) / c_phi) - Type(0.5);

    return xi;
}



// Templated version of GEV log-density
template<class Type>
Type gev_lpdf(Type x, Type mu, Type sigma, Type xi) {
    const Type tol = Type(1e-8);
    Type z = (x - mu) / sigma;

    if (CppAD::abs(xi) < tol) {
        return -log(sigma) - z - exp(-z);
    } else {
        Type t = Type(1.0) + xi * z;
        if (t <= Type(0.0)) {
            return -INFINITY;
        }
        Type inv_xi = Type(1.0) / xi;
        return -log(sigma) - (inv_xi + Type(1.0)) * log(t) - exp(-inv_xi * log(t));
    }
}

// Templated version of GEV CDF
template<class Type>
Type gev_cdf(Type x, Type mu, Type sigma, Type xi) {
    const Type tol = Type(1e-8);
    if (sigma <= Type(0.0)) return NAN;

    Type z = (x - mu) / sigma;

    // Gumbel limit ξ → 0
    if (CppAD::abs(xi) < tol)
        return exp(-exp(-z));

    // General ξ ≠ 0
    Type t = Type(1.0) + xi * z;
    if (t <= Type(0.0))  // outside support
        return (xi > Type(0.0) ? Type(0.0) : Type(1.0));

    Type inv_xi = Type(1.0) / xi;
    return exp(-exp(-inv_xi * log(t)));
}

// Templated version of t-distribution quantile with df = 4
template<class Type>
Type qt_df4(Type p) {
   Type alpha = 4 * p * (1 - p);
   Type q = cos(1/3 * acos(sqrt(alpha))) / sqrt(alpha);
   Type sign = (p < 0.5) ? Type(-1.0) : Type(1.0);
   Type Q = sign * 2 * sqrt(q - 1);
   return Q;
}

// GEV pdf for multiple stations
/**
 * @brief Main objective function implementing the multivariate GEV model
 * 
 * Combines:
 * 1. Marginal GEV distributions for each station
 * 2. Gaussian copula for spatial dependence
 * 3. Prior distributions for parameters
 * 
 * The negative log-likelihood is computed as:
 * - Sum of marginal GEV log-densities
 * - Gaussian copula contribution
 * - Prior contributions
 * 
 * @return Type Negative log-likelihood value
 */
template<class Type>
Type objective_function<Type>::operator() ()
{
    using namespace density;
    // Data
    DATA_MATRIX(y);          // Observed data
    DATA_SPARSE_MATRIX(Q);   // Precision matrix
    DATA_SCALAR(df);         // Degrees of freedom for t-distribution
    DATA_SCALAR(log_det_Q);  // Log-determinant of precision matrix
    // Priors
    DATA_VECTOR(prior_mu);   // Mean of priors for GEV parameters
    DATA_VECTOR(prior_sigma); // SD of priors for GEV parameters

    // Parameters
    PARAMETER_VECTOR(psi);   // location parameters 
    PARAMETER_VECTOR(tau);   // log-scale parameters
    PARAMETER_VECTOR(phi);   // shape parameters
    
    int N = y.rows();        // number of observations
    int P = y.cols();        // number of stations

    Type loglik = 0.0;
    vector<Type> z_i(P);

    // Calculate constants for xi_link function once
    const Type c_phi = Type(0.8);
    Type inside_term = Type(1.0) - pow(Type(0.5), c_phi);
    Type b_phi = -pow(c_phi, -1) * log(inside_term) * inside_term * pow(Type(2), c_phi - Type(1.0));
    Type a_phi = -b_phi * log(-log(inside_term));
    
    vector<Type> mu(P), sigma(P), xi(P);
    for(int p = 0; p < P; p++) {
        mu(p) = mu_link(psi(p)); 
        sigma(p) = sigma_link(psi(p), tau(p));
        xi(p) = xi_link(phi(p), a_phi, b_phi, c_phi);
    }
    
    // Loop over observations
    for(int i = 0; i < N; i++) {
        
        // Process all stations for this time point
        bool valid_observation = true;
        for(int p = 0; p < P; p++) {
            // Skip or handle NaN/Inf in the data
            if (!R_FINITE(asDouble(y(i, p)))) {
                valid_observation = false;
                break;
            }
            
            // Add GEV log-likelihood contribution with robustness
            Type log_f = gev_lpdf(y(i, p), mu(p), sigma(p), xi(p));
            if (!R_FINITE(asDouble(log_f))) {
                valid_observation = false;
                break;
            }
            loglik += log_f;
            
            // Transform to normal with bounds checking
            Type u = gev_cdf(y(i, p), mu(p), sigma(p), xi(p));
            // Check if u is very close to boundaries
            if (u < Type(1e-4)) {
                u = Type(1e-4);
            } else if (u > Type(1) - Type(1e-4)) {
                u = Type(1) - Type(1e-4);
            }
            Type z = qt_df4(u);
            
            // Check for valid z value
            if (!R_FINITE(asDouble(z)) || CppAD::abs(z) > Type(8)) {
                z = (u < Type(0.5) ? Type(-8.0) : Type(8.0));
            }
            z_i(p) = z;
        }
        
        Type quadform = -GMRF(Q, 1, false)(z_i);

        Type copula_contrib = 0.5 * log_det_Q - 0.5 * (df + P) * log(1 + quadform / df);
        // Subtract univariate normal log-densities
        for(int p = 0; p < P; p++) {
            copula_contrib += dt(z_i(p), df, true);
        }
        loglik += copula_contrib;
    }
    
    // Add priors
    for(int p = 0; p < P; p++) {
      Type prior_mu_p = dnorm(psi(p), prior_mu(0), prior_sigma(0), true);
      Type prior_tau_p = dnorm(tau(p), prior_mu(1), prior_sigma(1), true);
      Type prior_phi_p = dnorm(phi(p), prior_mu(2), prior_sigma(2), true);
      loglik += prior_mu_p;
      loglik += prior_tau_p;
      loglik += prior_phi_p;
    }

    Type nll = -loglik; 
    
    return nll;
}
