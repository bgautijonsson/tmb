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
  return std::exp( -std::pow(t, -inv_xi) );
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

    /* gradients vanish at F = 0 or 1, or if NaN/Inf */
    if (F <= Type(0) || F >= Type(1) || !R_FINITE(asDouble(F))) {
      px[0] = px[1] = px[2] = px[3] = Type(0);
      return;
    }

    Type z = (x - mu) / sigma;

    /* -------------- Gumbel (|ξ| < tol) --------------------------- */
    if (CppAD::abs(xi) < tol) {
      Type p = CppAD::exp(-z);                        // exp(-z)
      px[0] =  w * F * p / sigma;              // ∂F/∂x
      px[1] = -px[0];                          // ∂F/∂μ
      px[2] =  w * (-F * p * z / sigma);       // ∂F/∂σ
      px[3] = Type(0);                         // ∂F/∂ξ  (limit)
      return;
    }

    /* -------------- General ξ ≠ 0 ------------------------------- */
    Type t      = Type(1) + xi * z;
    Type inv_xi = Type(1) / xi;
    Type u      = CppAD::pow(t, -inv_xi);             // t^(-1/ξ)
    Type p      = u / t;                       // t^(-1/ξ - 1)

    /* ∂F/∂x */
    px[0] = w * F * p / sigma;

    /* ∂F/∂μ */
    px[1] = -px[0];

    /* ∂F/∂σ */
    px[2] = w * (-F * p * z / sigma);

    /* ∂F/∂ξ */
    px[3] = w * (-F * ( u * CppAD::log(t) / (xi*xi) - z * p / xi ));
  }
)

/* ------------------------------------------------------------------ */
/* Inline wrapper (AD aware) – call inside your objective function    */
/* ------------------------------------------------------------------ */
template<class Type>
Type pgev(const Type& x,
          const Type& mu,
          const Type& sigma,
          const Type& xi)
{
  if (sigma <= Type(0)) return NAN;       // cheap guard
  CppAD::vector<Type> tx(4);
  tx[0] = x;   tx[1] = mu;   tx[2] = sigma;   tx[3] = xi;
  return pgev(tx)[0];
}