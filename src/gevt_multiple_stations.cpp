#include <TMB.hpp>

// Link function for location parameter (mu)
template<class Type>
Type mu_link(Type psi) {
    return exp(psi);
}

// Link function for scale parameter (sigma)
template<class Type>
Type sigma_link(Type psi, Type tau) {
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

// Link function for trend parameter (Delta)
template<class Type>
Type delta_link(Type gamma) {
    return exp(gamma);
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
        return -log(sigma) - (inv_xi + Type(1.0)) * log(t) - CppAD::pow(t, -inv_xi);
    }
}

// GEV pdf for multiple stations
template<class Type>
Type objective_function<Type>::operator() ()
{
    DATA_MATRIX(y);          // Observed data
    DATA_MATRIX(X);          // Observed mean temperature
    DATA_VECTOR(prior_mu);   // Mean of priors for GEV parameters
    DATA_VECTOR(prior_sigma); // SD of priors for GEV parameters

    PARAMETER_VECTOR(psi);   // location parameters 
    PARAMETER_VECTOR(tau);   // log-scale parameters
    PARAMETER_VECTOR(phi);   // shape parameters
    PARAMETER_VECTOR(gamma); // trend parameters
    
    int N = y.rows();        // number of observations
    int P = y.cols();        // number of stations

    Type nll = 0.0;

    // Calculate constants for xi_link function once
    const Type c_phi = Type(0.8);
    Type inside_term = Type(1.0) - pow(Type(0.5), c_phi);
    Type b_phi = -pow(c_phi, -1) * log(inside_term) * inside_term * pow(Type(2), c_phi - Type(1.0));
    Type a_phi = -b_phi * log(-log(inside_term));
    
    // Loop over stations
    for(int p = 0; p < P; p++) {
        // Apply link functions
        Type mu0 = mu_link(psi(p));
        Type sigma = sigma_link(psi(p), tau(p));
        Type xi = xi_link(phi(p), a_phi, b_phi, c_phi);
        Type delta = delta_link(gamma(p));
        // Loop over observations
        for(int i = 0; i < N; i++) {
            Type mu = mu0 * (1 + delta * X(i, p));
            Type log_f = gev_lpdf(y(i, p), mu, sigma, xi);
            
            // Handle invalid values
            if(!R_FINITE(asDouble(log_f))) {
                nll = INFINITY;
                return nll;
            }
            
            nll -= log_f;
        }
        
        // Priors for station's parameters
        Type log_prior_psi = dnorm(psi(p), prior_mu(0), prior_sigma(0), true);
        Type log_prior_tau = dnorm(tau(p), prior_mu(1), prior_sigma(1), true);
        Type log_prior_phi = dnorm(phi(p), prior_mu(2), prior_sigma(2), true);
        Type log_prior_gamma = dnorm(gamma(p), prior_mu(3), prior_sigma(3), true);
        nll -= log_prior_psi;
        nll -= log_prior_tau;
        nll -= log_prior_phi;
        nll -= log_prior_gamma;
    }
    
    return nll;
}
