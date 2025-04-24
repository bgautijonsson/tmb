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
Type xi_link(Type phi) {
    Type c_phi = 0.8;
    Type b_phi = 0.39563;
    Type a_phi = 0.062376;
    
    Type xi_temp = (phi - a_phi) / b_phi;
    xi_temp = 1.0 - exp(-exp(xi_temp));
    return pow(xi_temp, 1.0/c_phi) - 0.5;
}

// GEV pdf for one station using atomic dgev
template<class Type>
Type objective_function<Type>::operator() ()
{
    DATA_VECTOR(y);          // Observed data
    PARAMETER(psi);          // location parameters (transformed)
    PARAMETER(tau);          // log-scale parameters (transformed)
    PARAMETER(phi);          // shape parameters (transformed)

    int n = y.size();        // number of observations

    Type nll = 0.0;

    // Transform parameters
    Type mu = mu_link(psi);
    Type sigma = sigma_link(psi, tau);
    Type xi = xi_link(phi);

    // Check sigma validity *once* before the loop
    if (sigma <= Type(0.0)) {
        // You could return INFINITY or handle appropriately
        // For consistency with the dgev wrapper, maybe return INFINITY
        return INFINITY;
    }

    // Vectorized version could be used here for efficiency
    for(int i = 0; i < n; i++) {
        // Calculate log-density using the atomic function wrapper
        Type log_f = dgev(y(i), mu, sigma, xi);

        // Check if the atomic function returned -Inf (e.g., y(i) outside support)
        if (!R_FINITE(asDouble(log_f))) {
            // If any observation is outside the support, the total likelihood is 0
            // -> negative log-likelihood is Inf
            return INFINITY; // Return immediately
        }

        nll -= log_f;
    }

    return nll;
}
