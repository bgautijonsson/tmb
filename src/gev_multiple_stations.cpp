#include <TMB.hpp>

// Scalar version of GEV log-density (for double type)
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
        return -log(sigma) - (inv_xi + 1.0) * log(t) - pow(t, -inv_xi);
    }
}

// Define atomic GEV function
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
            Type pow_t_term = pow(t, -inv_xi - Type(1.0));

            px[0] = w * (pow_t_term - (xi + Type(1.0)) / t) / sigma;
            px[1] = -px[0]; 
            px[2] = w * (Type(-1.0) + (xi + Type(1.0)) * z / t - z * pow_t_term) / sigma;
            
            Type log_t = log(t);
            px[3] = w * ( 
                    (Type(1.0) - pow(t, -inv_xi)) * log_t / (xi*xi) 
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


// GEV pdf for multiple stations
template<class Type>
Type objective_function<Type>::operator() ()
{
    DATA_MATRIX(y);          // Observed data
    DATA_VECTOR(prior_mu);   // Mean of priors for GEV parameters
    DATA_VECTOR(prior_sigma); // SD of priors for GEV parameters
    
    PARAMETER_VECTOR(psi);   // location parameters 
    PARAMETER_VECTOR(tau);   // log-scale parameters
    PARAMETER_VECTOR(phi);   // shape parameters
    
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
        Type mu = mu_link(psi(p));
        Type sigma = sigma_link(psi(p), tau(p));
        Type xi = xi_link(phi(p), a_phi, b_phi, c_phi);
    
        // Loop over observations
        for(int i = 0; i < N; i++) {
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
        nll -= log_prior_psi;
        nll -= log_prior_tau;
        nll -= log_prior_phi;
    }
    
    return nll;
}
