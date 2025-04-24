/**
 * @file gev_multiple_stations_with_copula.cpp
 * @brief Implementation of a multivariate GEV model with Gaussian copula dependence
 * 
 * This file implements a multivariate Generalized Extreme Value (GEV) model for multiple stations
 * with spatial dependence modeled through a Gaussian copula. The implementation uses TMB 
 * (Template Model Builder) for automatic differentiation and parameter estimation.
 * 
 * The model consists of:
 * 1. Marginal GEV distributions for each station
 * 2. Spatial dependence modeled via a Gaussian copula
 * 3. Parameter transformations to ensure valid parameter spaces
 * 
 * Mathematical Background:
 * - GEV Distribution: F(x|μ,σ,ξ) = exp{-[1 + ξ((x-μ)/σ)]^(-1/ξ)}
 * - Parameters:
 *   μ (location): μ = exp(ψ)
 *   σ (scale): σ = exp(ψ + τ)
 *   ξ (shape): ξ = g(φ) where g is a bounded transformation
 */

#include <TMB.hpp>

// Scalar version of GEV log-density (for double type)
/**
 * @brief Computes the GEV log-density for scalar inputs
 * 
 * @param x Value at which to evaluate the density
 * @param mu Location parameter
 * @param sigma Scale parameter (must be positive)
 * @param xi Shape parameter
 * @return double Log-density value
 * 
 * @note Special handling for xi near zero (Gumbel case)
 */
double dgev_scalar(double x, double mu, double sigma, double xi) {
    const double tol = 1e-8;
    double z = (x - mu) / sigma;

    if (fabs(xi) < tol) {
        return -log(sigma) - z - exp(-z);
    } else {
        double t = 1.0 + xi * z;
        if (t <= 0.0) {
            return -INFINITY;
        }
        double inv_xi = 1.0 / xi;
        return -log(sigma) - (inv_xi + 1.0) * log(t) - exp(-inv_xi * log(t));
    }
}

// Define atomic GEV function
/**
 * @brief TMB atomic function for GEV density with automatic differentiation
 * 
 * Implements forward and reverse mode automatic differentiation for the GEV density.
 * This enables efficient computation of derivatives needed for optimization.
 */
TMB_ATOMIC_VECTOR_FUNCTION(
    // ATOMIC_NAME
    dgev,
    // OUTPUT_DIM
    1,
    // ATOMIC_DOUBLE
    ty[0] = dgev_scalar(tx[0], tx[1], tx[2], tx[3]);
    ,
    // ATOMIC_REVERSE
    Type x = tx[0];
    Type mu = tx[1];
    Type sigma = tx[2];
    Type xi = tx[3];
    Type w = py[0];  // Adjoint from upstream

    // If result was -Inf, all derivatives are zero
    if (!R_FINITE(asDouble(ty[0]))) {
        px[0] = Type(0.0);
        px[1] = Type(0.0);
        px[2] = Type(0.0);
        px[3] = Type(0.0);
    } else {
        Type z = (x - mu) / sigma;
        Type tol = Type(1e-8);

        if (CppAD::abs(xi) < tol) {
            // Gumbel case
            Type exp_neg_z = exp(-z);
            px[0] = w * (exp_neg_z - Type(1.0)) / sigma;
            px[1] = -px[0];
            px[2] = w * (Type(-1.0) + z - z * exp_neg_z) / sigma;
            px[3] = Type(0.0);
        } else {
            // General GEV case
            Type t = Type(1.0) + xi * z;
            Type inv_xi = Type(1.0) / xi;
            Type pow_t_term = exp((-inv_xi - Type(1.0)) * log(t));

            px[0] = w * (pow_t_term - (xi + Type(1.0)) / t) / sigma;
            px[1] = -px[0]; 
            px[2] = w * (Type(-1.0) + (xi + Type(1.0)) * z / t - z * pow_t_term) / sigma;
            
            Type log_t = log(t);
            px[3] = w * ( 
                    (Type(1.0) - exp(-inv_xi * log(t))) * log_t / (xi*xi)
                    - (Type(1.0) + inv_xi) * z / t
                    + z * pow_t_term / xi
                );
        }
    }
)

// Scalar wrapper function
template<class Type>
Type dgev(const Type &x, const Type &mu, const Type &sigma, const Type &xi) {
    if (sigma <= Type(0.0)) {
        return -INFINITY;
    }
    CppAD::vector<Type> tx(4);
    tx[0] = x;
    tx[1] = mu;
    tx[2] = sigma;
    tx[3] = xi;
    return dgev(tx)[0];
}

/* ------------------------------------------------------------------ */
/* Convenience: double-only scalar version (unit tests, debugging)    */
/* ------------------------------------------------------------------ */
double pgev_scalar(double x, double mu, double sigma, double xi)
{
    const double tol = 1e-8;
    if (sigma <= 0.0) return NAN;

    double z = (x - mu) / sigma;

    /* ----- Gumbel limit ξ → 0 -------------------------------------- */
    if (std::fabs(xi) < tol)
        return std::exp( -std::exp(-z) );

    /* ----- General ξ ≠ 0 ------------------------------------------- */
    double t = 1.0 + xi * z;
    if (t <= 0.0)                      // outside support
        return (xi > 0.0 ? 0.0 : 1.0);

    double inv_xi = 1.0 / xi;
    return std::exp( -std::exp(-inv_xi * std::log(t)) );
}

/* ------------------------------------------------------------------ */
/* Atomic definition (vector style)                                   */
/* ------------------------------------------------------------------ */
TMB_ATOMIC_VECTOR_FUNCTION(
  /* ATOMIC NAME  */ pgev,
  /* OUTPUT DIM   */ 1,

    /* ========== forward mode for double ========== */
    ty[0] = pgev_scalar(tx[0], tx[1], tx[2], tx[3]);
    ,
    /* ========== reverse mode (back-prop) ========== */
    {
        Type x     = tx[0];
        Type mu    = tx[1];
        Type sigma = tx[2];
        Type xi    = tx[3];

        Type w = py[0];                // upstream adjoint
        Type F = ty[0];                // CDF just computed

        const Type tol = Type(1e-8);

        /* ---- cases where gradient is identically zero ---- */
        if (F <= Type(0) || F >= Type(1) || !R_FINITE(asDouble(F))) {
            px[0] = px[1] = px[2] = px[3] = Type(0);
        }       

        Type z = (x - mu) / sigma;

        if (CppAD::abs(xi) < tol) {
            /* Gumbel */
        Type p = exp(-z);
            px[0] =  w * F * p / sigma;
            px[1] = -px[0];
            px[2] =  w * (-F * p * z / sigma);
            px[3] = Type(0);
        } else {
            /* General xi */
            Type t  = Type(1) + xi * z;
            Type inv= Type(1) / xi;
            Type u  = exp(-inv * log(t));
            Type p  = u / t;

            px[0] = w * F * p / sigma;
            px[1] = -px[0];
            px[2] = w * (-F * p * z / sigma);
            px[3] = w * (-F * ( u * log(t)/(xi*xi) - z * p / xi ));
        }
    }
)

/* ------------------------------------------------------------------ */
/* Inline wrapper (AD aware) – call inside your objective function    */
/* ------------------------------------------------------------------ */
template<class Type>
Type pgev(const Type& x, const Type& mu, const Type& sigma, const Type& xi) {
    if (sigma <= Type(0)) return Type(1e-12);  // Small constant instead of negative value
    
    Type z = (x - mu) / sigma;
    const Type tol = Type(1e-8);
    Type result;
    
    // Guard against extreme values that could cause numerical issues
    if (CppAD::abs(z) > Type(50)) {
        return (z > Type(0) ? Type(1) - Type(1e-12) : Type(1e-12));
    }
    
    if (CppAD::abs(xi) < tol) {
        // Gumbel case - more stable implementation
        Type e = exp(-z);
        if (e > Type(100)) return Type(1e-12);
        result = exp(-e);
    } else {
        // General GEV case
        Type t = Type(1.0) + xi * z;
        if (t <= Type(0)) {
            return (xi > Type(0) ? Type(1e-12) : Type(1) - Type(1e-12));
        }
        Type inv_xi = Type(1.0) / xi;
        result = exp(-exp(-inv_xi * log(t)));
    }
    
    // Clamp to avoid exact 0 or 1
    Type eps = Type(1e-12);
    if (result <= Type(0)) return eps;
    if (result >= Type(1)) return Type(1) - eps;
    return result;
}


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
Type sigma_link(Type psi, Type tau) {
    return exp(psi + tau);
}

// Link function for shape parameter (xi)
/**
 * @brief Link function for GEV shape parameter
 * 
 * Transforms unconstrained parameter φ to shape parameter ξ using a bounded
 * transformation that ensures numerical stability
 * 
 * @param phi Unconstrained parameter
 * @return Type Transformed shape parameter
 */
template<class Type>
Type xi_link(Type phi) {
    Type c_phi = 0.8;
    Type b_phi = 0.39563;
    Type a_phi = 0.062376;
    
    Type xi_temp = (phi - a_phi) / b_phi;
    xi_temp = 1.0 - exp(-exp(xi_temp));
    return pow(xi_temp, 1.0/c_phi) - 0.5;
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
    
    vector<Type> mu(P), sigma(P), xi(P);
    for(int p = 0; p < P; p++) {
        mu(p) = mu_link(psi(p));  // Record once
        sigma(p) = sigma_link(psi(p), tau(p));
        xi(p) = xi_link(phi(p));
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
            Type log_f = dgev(y(i, p), mu(p), sigma(p), xi(p));
            if (!R_FINITE(asDouble(log_f))) {
                valid_observation = false;
                break;
            }
            loglik += log_f;
            
            // Transform to normal with bounds checking
            Type u = pgev(y(i, p), mu(p), sigma(p), xi(p));
            // Check if u is very close to boundaries
            if (u < Type(1e-4)) {
                u = Type(1e-4);
            } else if (u > Type(1) - Type(1e-4)) {
                u = Type(1) - Type(1e-4);
            }
            Type z = qnorm(u);
            
            // Check for valid z value
            if (!R_FINITE(asDouble(z)) || CppAD::abs(z) > Type(8)) {
                z = (u < Type(0.5) ? Type(-8.0) : Type(8.0));
            }
            z_i(p) = z;
        }
        
        Type copula_contrib = -GMRF(Q)(z_i);
        // Subtract univariate normal log-densities
        for(int p = 0; p < P; p++) {
            copula_contrib += dnorm(z_i(p), Type(0), Type(1), true);
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
