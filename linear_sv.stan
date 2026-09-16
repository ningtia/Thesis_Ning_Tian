// linear_sv.stan
// The null model: the same stochastic-volatility specification as
// nonlinear_sv_v3.stan with the neural-network term removed (g == 0).
//
// Purpose: model comparison. Requiring the NN model to sample cleanly when the
// true amplitude is zero is asking NUTS to explore a degenerate posterior --
// tau_w -> 0 leaves W1 and b1 unidentified, and the funnel sharpens as T grows
// (your null scenario gives 0/10 usable fits at T = 1000). Comparing this model
// against the NN model by LOO answers "is there a nonlinearity?" without ever
// sampling that degenerate region.
//
// 2026-08-25: X is no longer a dummy. This now carries the real benchmark
// covariates (default_benchmark_covariates() in benchmark_utils.R --
// GCPU_baseline, l_t, c_t, itraxx_robustness, Term_Spread, Brent_return,
// TTF_return, log_VSTOXX, COVID_dummy, Energy_crisis_dummy), entering the log-
// volatility transition exactly the way regime_switching_sv.stan's do (same
// X[t] * beta term, same beta ~ normal(0, 1) prior, contemporaneous T rows
// instead of nonlinear_sv_v3.stan's T-1). That is deliberate: linearSV is now
// regimeSwitchingSV's single-regime twin -- everything is identical except the
// regime switching itself -- so a QLIKE/LPS gap between the two can be
// attributed to the regime structure alone, not to one model seeing the
// covariates and the other not. See linear_sv.R's forecast_linear_sv_one_step()
// for the matching out-of-sample side of this (mirrors
// forecast_ms_sv_one_step()'s x_effect term).
//
// K, s_fixed, tau_w_scale, b1_scale stay as unused placeholders: nothing here
// depends on removing them, and this file no longer needs to share a schema
// with results/simulation/'s copy of linear_sv.stan (a separate file used only
// by the simulation-study scripts there).

data {
    int<lower=2> T;
    int<lower=1> D;
    int<lower=1> K;                          // unused here; kept for a shared data list
    vector[T] y;
    matrix[T, D] X;                          // real covariates, contemporaneous with h
    int<lower=0, upper=1> use_student_t;
    vector[D] x_forecast;                    // covariate row one step past the training window

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
    vector[D] beta;
    vector[T] eta_raw;
    array[use_student_t ? 1 : 0] real<lower=0> nu_minus2;
}

transformed parameters {
    real phi = 2 * phi_star - 1;
    vector[T] h;

    h[1] = mu + X[1] * beta + (stationary_init
                 ? sigma_eta / sqrt(1 - square(phi))
                 : h1_scale) * eta_raw[1];
    for (t in 2:T)
        h[t] = mu + phi * (h[t-1] - mu) + X[t] * beta + sigma_eta * eta_raw[t];
}

model {
    mu        ~ normal(0, mu_scale);
    phi_star  ~ beta(phi_a, phi_b);
    sigma_eta ~ normal(0, sigma_eta_scale);
    beta      ~ normal(0, 1);
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

    h_forecast   = mu + phi * (h[T] - mu) + dot_product(x_forecast, beta) + sigma_eta * normal_rng(0, 1);
    vol_forecast = exp(0.5 * h_forecast);
}
