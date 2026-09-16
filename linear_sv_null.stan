// linear_sv_null.stan
//
// The genuine nested null of nonlinear_sv.stan: the same SV backbone with the
// neural-network term removed (g == 0 identically -- not estimated toward
// zero, structurally absent). Priors on every shared parameter (mu, phi_star,
// sigma_eta, nu_minus2) are the same distributions referencing the same named
// hyperparameters nonlinear_sv.stan's data block takes, so as long as the
// caller passes the same stan_data list to both models (mu_scale, phi_a,
// phi_b, sigma_eta_scale, nu_rate identical), this model sits exactly where
// nonlinear_sv.stan lands when its NN output g is forced to zero. That is
// what makes a LOO/DM comparison between the two answer "is there a
// nonlinearity" rather than "did we also change the SV priors."
//
// 2026-08-25: split out from linear_sv.stan, which now carries real exogenous
// covariates (X * beta, matching regime_switching_sv.stan) for its role as
// regimeSwitchingSV's single-regime twin in the run_models.R benchmark table.
// That covariate term has no counterpart in nonlinear_sv.stan (there X only
// ever reaches h through the NN, and no set of NN weights reproduces a linear
// X * beta term), so linear_sv.stan stopped being nested the moment it grew
// beta. This file is the one to use wherever the null model needs to be
// nonlinear_sv.stan's exact nested null (e.g. a real-data counterpart of
// results/simulation/null_test_loo.R's LOO comparison) -- not
// run_models.R's benchmark table, which should keep using linear_sv.stan.
//
// The data block is deliberately identical to nonlinear_sv.stan's, including
// the NN-only arguments (K, X, x_forecast, s_fixed, tau_w_scale, b1_scale),
// so the exact same stan_data list can be passed to both models unmodified --
// this file just never reads the NN-only fields.

data {
    int<lower=2> T;
    int<lower=1> D;
    int<lower=1> K;                          // unused here; kept for a shared data list
    vector[T] y;
    matrix[T-1, D] X;                        // unused here (same shape as nonlinear_sv.stan's)
    int<lower=0, upper=1> use_student_t;
    vector[D] x_forecast;                    // unused here

    real<lower=0> s_fixed;                   // unused here
    real<lower=0> tau_w_scale;               // unused here

    real mu_scale;
    real<lower=0> phi_a;
    real<lower=0> phi_b;
    real<lower=0> sigma_eta_scale;
    real<lower=0> b1_scale;                  // unused here
    real<lower=0> nu_rate;

    int<lower=0, upper=1> prior_only;
    int<lower=0, upper=1> stationary_init;
    real<lower=0> h1_scale;
}

parameters {
    real mu;
    real<lower=0, upper=1> phi_star;
    real<lower=0> sigma_eta;
    vector[T] eta_raw;
    array[use_student_t ? 1 : 0] real<lower=0> nu_minus2;
}

transformed parameters {
    real phi = 2 * phi_star - 1;
    vector[T] h;

    h[1] = mu + (stationary_init
                 ? sigma_eta / sqrt(1 - square(phi))
                 : h1_scale) * eta_raw[1];
    for (t in 2:T)
        h[t] = mu + phi * (h[t-1] - mu) + sigma_eta * eta_raw[t];
}

model {
    mu        ~ normal(0, mu_scale);
    phi_star  ~ beta(phi_a, phi_b);
    sigma_eta ~ normal(0, sigma_eta_scale);
    eta_raw   ~ std_normal();

    if (use_student_t)
        nu_minus2 ~ gamma(2, nu_rate);

    if (!prior_only) {
        if (use_student_t) {
            real nu = nu_minus2[1] + 2;
            y ~ student_t(nu, 0, exp(0.5 * h));
        } else {
            y ~ normal(0, exp(0.5 * h));
        }
    }
}

generated quantities {
    vector[T] log_lik;
    real h_forecast;
    real vol_forecast;
    real nu = use_student_t ? nu_minus2[1] + 2 : positive_infinity();
    real h_bar = mean(h);

    if (use_student_t) {
        for (t in 1:T)
            log_lik[t] = student_t_lpdf(y[t] | nu, 0, exp(0.5 * h[t]));
    } else {
        for (t in 1:T)
            log_lik[t] = normal_lpdf(y[t] | 0, exp(0.5 * h[t]));
    }

    h_forecast   = mu + phi * (h[T] - mu) + sigma_eta * normal_rng(0, 1);
    vol_forecast = exp(0.5 * h_forecast);
}
