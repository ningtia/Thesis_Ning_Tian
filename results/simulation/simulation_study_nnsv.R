# simulation_study_nnsv_v2.R
#
# Simulate observations from a nonlinear stochastic-volatility model.
#
#   y_t = exp(h_t / 2) * eps_t
#   h_t = mu + phi (h_{t-1} - mu) + g_t + sigma_eta * eta_t,   eta_t ~ N(0,1)
#   g_t = g_scale * sin(x_{t-1})  [+ lev_scale * min(y_{t-1}, 0) if enabled]
#
# simulate_nnsv()'s defaults (x_ar = 0, standardise_t = FALSE, burn_in = 0,
# lev_scale = 0) reproduce the very first, i.i.d.-covariate version of this
# DGP; none of the designs below use them as-is. A persistent x_ar and a
# burn-in are what make g_t identifiable at this T (see design (a) below),
# so every design generated here overrides them. The designs actually used
# are the calls below.

simulate_nnsv <- function(seed          = 20260820,
                          T_sim         = 500,
                          mu            = 1.80,
                          phi           = 0.95,
                          sigma_eta     = 0.25,
                          nu            = 10,
                          g_scale       = 0.25,
                          lev_scale     = 0.00,   # coefficient on y_{t-1}^-
                          x_ar          = 0.00,   # persistence of the covariate
                           standardise_t = FALSE,  # FALSE keeps raw Student-t innovations
                          burn_in       = 0) {

  stopifnot(abs(phi) < 1, abs(x_ar) < 1, sigma_eta > 0, nu > 2)
  set.seed(seed)

  n <- burn_in + T_sim

  # --- exogenous covariate: i.i.d. if x_ar = 0, AR(1) with unit variance else
  if (x_ar == 0) {
    x <- rnorm(n)
  } else {
    x <- numeric(n)
    x[1] <- rnorm(1)                                  # stationary sd = 1
    innov <- rnorm(n) * sqrt(1 - x_ar^2)
    for (t in 2:n) x[t] <- x_ar * x[t - 1] + innov[t]
  }

  eta <- rnorm(n)

  # --- observation innovations.  With the raw rt() scale, exp(h) is the
  #     squared scale and Var(y | h) = exp(h) * nu / (nu - 2).
  epsilon <- rt(n, df = nu)
  if (standardise_t) epsilon <- epsilon / sqrt(nu / (nu - 2))

  h <- numeric(n); y <- numeric(n); g <- numeric(n)

  # h[1] from the AR(1) stationary law.  With burn_in > 0 this only sets the
  # starting point of a chain that is stationary by the time it is kept, so the
  # slight understatement of the variance (g contributes too) does not matter.
  h[1] <- mu + (sigma_eta / sqrt(1 - phi^2)) * eta[1]
  y[1] <- exp(h[1] / 2) * epsilon[1]

  for (t in 2:n) {
    g[t] <- g_scale * sin(x[t - 1]) + lev_scale * min(y[t - 1], 0)
    h[t] <- mu + phi * (h[t - 1] - mu) + g[t] + sigma_eta * eta[t]
    y[t] <- exp(h[t] / 2) * epsilon[t]
  }

  keep <- (burn_in + 1):n
  data.frame(
    time   = seq_len(T_sim),
    y      = y[keep],
    x      = x[keep],
    h_true = h[keep],
    g_true = g[keep]
  )
}

# ---------------------------------------------------------------------------
# Derived quantities that the recovery table should be compared against.
# g_scale itself is NOT a parameter of the estimated model and cannot be
# recovered; sd(g) is its estimable counterpart
# ---------------------------------------------------------------------------
dgp_facts <- function(d, sigma_eta = 0.25, phi = 0.95, nu = 10,
                      standardise_t = FALSE) {
  g <- d$g_true[-1]
  data.frame(
    quantity = c("sd(g)",
                 "sd of innovation if g is missed",
                 "stationary sd of h",
                 "sample mean of g",
                 "level shift implied by that mean",
                 "Var(y|h) / exp(h)"),
    value = c(sd(g),
              sqrt(sigma_eta^2 + var(g)),
              sd(d$h_true),
              mean(g),
              mean(g) / (1 - phi),
              if (standardise_t) 1 else nu / (nu - 2))
  )
}

# ---------------------------------------------------------------------------
# (a) Recommended design: persistent covariate, raw Student-t innovations,
#     burn-in.  A persistent x makes g_t persistent, which i.i.d. state noise
#     cannot mimic -- this is what makes the nonlinear term identifiable at
#     T = 312.  This raw Student-t convention matches the Stan likelihood.
# ---------------------------------------------------------------------------
sim_main <- simulate_nnsv(
  seed = 20260820, x_ar = 0.90, standardise_t = FALSE, burn_in = 200
)

# ---------------------------------------------------------------------------
# (b) Null DGP: no nonlinearity.  The network should shrink to zero here.
#     This is the false-positive check.
# ---------------------------------------------------------------------------
sim_null <- simulate_nnsv(
  seed = 20260820, g_scale = 0, x_ar = 0.90, standardise_t = FALSE, burn_in = 200
)

# ---------------------------------------------------------------------------
# (c) Strong, asymmetric DGP: g_sd deliberately far from the prior (prior_only
#     puts g_sd's median around 0.151, 90% interval roughly [0.012, 0.378]
#     with K=4, s_fixed=0.5, tau_w_scale=0.5 -- see prior_predictive_check.R).
#     The main design's g_sd (~0.16) sits close to that prior BY
#     CONSTRUCTION, so "posterior looks like prior" there is not evidence of
#     anything. This is the identifiability stress test advisor feedback
#     asked for: can the data pull the posterior away from the prior when
#     the truth genuinely isn't near it? The leverage term makes the shape
#     clearly asymmetric (only active when y_{t-1} < 0), not just bigger.
#     g_scale/lev_scale were picked by linear extrapolation from the main
#     design's realized sd(g); if the printed sd(g) below misses the
#     0.40-0.50 target, adjust them and rerun this block.
# ---------------------------------------------------------------------------
sim_strong <- simulate_nnsv(
  seed = 20260820, g_scale = 0.65, lev_scale = 0.15,
  x_ar = 0.90, standardise_t = FALSE, burn_in = 200
)
strong_g <- sim_strong$g_true[-1L]
message("strong design: realized sd(g) = ", round(sd(strong_g), 4),
        " (target 0.40-0.50); skewness = ",
        round(mean((strong_g - mean(strong_g))^3) / sd(strong_g)^3, 3),
        " (0 = symmetric, further from 0 = clearer asymmetry)")

strong_true_mu <- 1.80 + mean(strong_g) / (1 - 0.95)
message("strong design: true mu = ", round(strong_true_mu, 4),
        " (mu_dgp=1.80 + mean(g)/(1-phi)); true g_sd = ", round(sd(strong_g), 4),
        " -- consumed directly from nnsv_simulation_strong.csv's h_true/g_true",
        " columns by prior_predictive_check.R, no separate true-parameters",
        " CSV needed (unlike nnsv_main_true_parameters.csv, which",
        " Simulation_recovery.R reads on its own).")

main_g <- sim_main$g_true[-1L]
main_true_parameters <- data.frame(
  parameter = c("mu", "phi", "sigma_eta", "nu", "g_sd"),
  true_value = c(
    1.80 + mean(main_g) / (1 - 0.95),
    0.95,
    0.25,
    10,
    sd(main_g)
  ),
  generator_value = c(1.80, 0.95, 0.25, 10, 0.25),
  definition = c(
    "mu_dgp + mean(g_true[2:T]) / (1 - phi)",
    "AR(1) persistence",
    "state innovation standard deviation",
    "Student-t degrees of freedom",
    "sample sd(g_true[2:T]); generator_value is g_scale"
  )
)

write.csv(sim_main, "nnsv_simulation_main.csv",     row.names = FALSE)
write.csv(sim_null, "nnsv_simulation_null.csv",     row.names = FALSE)
write.csv(sim_strong, "nnsv_simulation_strong.csv", row.names = FALSE)
write.csv(main_true_parameters,"nnsv_main_true_parameters.csv", row.names = FALSE)

cat("\n-- recommended design --\n"); print(dgp_facts(sim_main, standardise_t = FALSE))
