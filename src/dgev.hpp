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