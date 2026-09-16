// nonlinear_sv_v3.stan
//
// v3 adds three things to v2, all aimed at the issues the 60-replication study
// exposed:
//   * prior_only      : drop the likelihood, so the prior predictive for g_sd
//                       (and everything else) can be compared with the
//                       posterior.  g_sd looks prior-dominated; check it.
//   * stationary_init : h[1] no longer uses sigma_eta / sqrt(1 - phi^2), which
//                       explodes as phi -> 1 and is the likely source of the
//                       stuck chains.  One observation carries almost no
//                       likelihood weight, so a fixed diffuse scale costs
//                       nothing.
//   * h_bar           : the sample level of the log-variance.  When phi -> 1,
//                       mu is not identified but h_bar still is, and h_bar is
//                       what the application actually cares about.
//
// Nonlinear stochastic volatility with a small neural-network transition term.
//
//   y_t   = exp(h_t / 2) * eps_t,          eps_t ~ Student-t(nu) (scale 1)
//   h_t   = mu + phi (h_{t-1} - mu) + g_t + sigma_eta * eta_t
//   g_t   = s_fixed * tanh( w2' tanh(W1 z_{t-1} + b1) ),  centred to mean zero
//   z_{t-1} = standardised [ y_{t-1}^+ , y_{t-1}^- , x_{t-1} ]
//
// Changes relative to nonlinear_sv.stan, and why:
//
//  (1) The NN output is centred to have exactly zero sample mean.  The intercept
//      of the transition is mu*(1-phi) + mean(g).  With phi ~ 0.95 a constant
//      offset of only 0.05 in g shifts mu by 1.0, so an uncentred network makes
//      mu essentially unidentified.  Centring assigns the level to mu alone.
//
//  (2) h_{t-1} is no longer an NN input.  With h inside the network the latent
//      path becomes a *recurrent* nonlinear function of the T auxiliary normals,
//      which is what produces the max-treedepth behaviour; and the network can
//      then re-create the mean-reversion term itself.  With observable-only
//      inputs, h is affine in eta_raw given the parameters and the forward pass
//      vectorises over t.  (See notes at the bottom for how to put h back.)
//
//  (3) The output bias b2 is dropped: after centring it only adds a flat ridge.
//
//  (4) tau_w scales the output layer only.  Previously W1 = tau_w * W1_raw and
//      w2 = tau_w * w2_raw, so the NN pre-activation was quadratic in tau_w,
//      creating a ridge and a funnel in (tau_w, W1_raw, w2_raw).
//
//  (5) w2 is a positive_ordered vector.  This removes the K! label-switching
//      modes (K = 8 gives 40320 equivalent modes) and the hidden-unit sign flip.
//
//  (6) Prior hyperparameters are passed as data so that prior sensitivity can be
//      reported rather than asserted.

data {
    int<lower=2> T;                          // number of time periods
    int<lower=1> D;                          // dimension of exogenous covariates
    int<lower=1> K;                          // number of hidden units
    vector[T] y;                             // observed returns (percent)
    matrix[T-1, D] X;                        // covariates x_{t-1}, rows t = 2..T
    int<lower=0, upper=1> use_student_t;     // 0 = Gaussian, 1 = Student-t
    vector[D] x_forecast;                    // x_T, for the one-step forecast

    real<lower=0> s_fixed;                   // bound on the NN contribution
    real<lower=0> tau_w_scale;               // half-normal scale for tau_w

    // prior hyperparameters (supplied so they can be varied)
    real mu_scale;                           // sd of the normal prior on mu
    real<lower=0> phi_a;                     // beta shape a for (phi+1)/2
    real<lower=0> phi_b;                     // beta shape b for (phi+1)/2
    real<lower=0> sigma_eta_scale;           // half-normal scale for sigma_eta
    real<lower=0> b1_scale;                  // normal scale for hidden biases
    real<lower=0> nu_rate;                   // gamma rate for nu - 2 (shape 2)

    int<lower=0, upper=1> prior_only;        // 1 = drop the likelihood
    int<lower=0, upper=1> stationary_init;   // 1 = stationary h[1] (as in v2)
    real<lower=0> h1_scale;                  // scale of h[1] when stationary_init = 0
}

transformed data {
    int N    = T - 1;                        // usable transitions, t = 2..T
    int D_nn = D + 2;                        // NN inputs: y+, y-, x

    matrix[N, D_nn] Z;                       // standardised NN inputs
    row_vector[D_nn] z_center;
    row_vector[D_nn] z_scale;
    row_vector[D_nn] z_fore_raw;             // unstandardised forecast input

    {
        matrix[N, D_nn] Z_raw;
        for (i in 1:N) {
            Z_raw[i, 1] = fmax(y[i], 0.0);
            Z_raw[i, 2] = fmin(y[i], 0.0);
            Z_raw[i, 3:D_nn] = X[i];
        }
        for (j in 1:D_nn) {
            z_center[j] = mean(Z_raw[, j]);
            z_scale[j]  = sd(Z_raw[, j]);
            if (!(z_scale[j] > 0)) z_scale[j] = 1.0;
            Z[, j] = (Z_raw[, j] - z_center[j]) / z_scale[j];
        }
        z_fore_raw[1] = fmax(y[T], 0.0);
        z_fore_raw[2] = fmin(y[T], 0.0);
        z_fore_raw[3:D_nn] = x_forecast';
    }
}

parameters {
    // --- SV parameters ---
    real mu;                                 // log-variance level
    real<lower=0, upper=1> phi_star;         // (phi + 1) / 2
    real<lower=0> sigma_eta;                 // log-variance innovation sd

    // --- NN parameters (one hidden layer, no output bias) ---
    matrix[K, D_nn] W1;                      // input-to-hidden weights
    vector[K] b1;                            // hidden biases
    positive_ordered[K] w2_raw;              // ordered => symmetry broken
    real<lower=0> tau_w;                     // scale of the output layer

    // --- latent path (non-centred) ---
    vector[T] eta_raw;

    // --- heavy tails (optional) ---
    array[use_student_t ? 1 : 0] real<lower=0> nu_minus2;
}

transformed parameters {
    real phi = 2 * phi_star - 1;
    vector[K] w2 = tau_w * w2_raw;

    vector[N] g;                             // centred NN contribution, t = 2..T
    real g_bar;                              // its removed mean (needed for GQ)
    vector[T] h;                             // latent log-variance path

    {
        matrix[N, K] A = tanh(Z * W1' + rep_matrix(b1', N));
        vector[N] g_raw = s_fixed * tanh(A * w2);
        g_bar = mean(g_raw);
        g = g_raw - g_bar;                   // level belongs to mu, not to g
    }

    // The stationary scale sigma_eta / sqrt(1 - phi^2) diverges as phi -> 1 and
    // couples phi, sigma_eta and eta_raw[1] into a funnel.  With
    // stationary_init = 0 the single initial state gets a fixed diffuse scale.
    h[1] = mu + (stationary_init
                 ? sigma_eta / sqrt(1 - square(phi))
                 : h1_scale) * eta_raw[1];
    for (t in 2:T)
        h[t] = mu + phi * (h[t-1] - mu) + g[t-1] + sigma_eta * eta_raw[t];
}

model {
    // === PRIORS ===
    mu        ~ normal(0, mu_scale);
    phi_star  ~ beta(phi_a, phi_b);
    sigma_eta ~ normal(0, sigma_eta_scale);      // half-normal

    to_vector(W1) ~ std_normal();
    b1            ~ normal(0, b1_scale);
    w2_raw        ~ std_normal();                // half-normal on ordered values
    tau_w         ~ normal(0, tau_w_scale);      // half-normal

    eta_raw ~ std_normal();

    if (use_student_t)
        nu_minus2 ~ gamma(2, nu_rate);           // nu = nu_minus2 + 2 > 2

    // === LIKELIHOOD ===
    // prior_only = 1 leaves the priors above in place and drops the data, so
    // the resulting draws ARE the prior predictive.  Compare the prior and
    // posterior densities of g_sd: if they coincide, the nonlinearity is not
    // identified by the data and must not be reported as recovered.
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
    real vol_forecast;                       // Student-t SCALE, exp(h_forecast/2)
    real sd_forecast;                        // actual conditional SD -- use this one
    real g_sd = sd(g);                       // amplitude of the fitted nonlinearity
    real nu = use_student_t ? nu_minus2[1] + 2 : positive_infinity();

    // Sample level of the log-variance.  Identified even when phi -> 1 and mu
    // is not; compare against mean(h_true).  Report this next to mu.
    real h_bar = mean(h);

    if (use_student_t) {
        for (t in 1:T)
            log_lik[t] = student_t_lpdf(y[t] | nu, 0, exp(0.5 * h[t]));
    } else {
        for (t in 1:T)
            log_lik[t] = normal_lpdf(y[t] | 0, exp(0.5 * h[t]));
    }

    {
        vector[D_nn] z_T;
        for (j in 1:D_nn)
            z_T[j] = (z_fore_raw[j] - z_center[j]) / z_scale[j];

        vector[K] a_T = tanh(W1 * z_T + b1);
        real g_T = s_fixed * tanh(dot_product(w2, a_T)) - g_bar;

        h_forecast   = mu + phi * (h[T] - mu) + g_T + sigma_eta * normal_rng(0, 1);
        vol_forecast = exp(0.5 * h_forecast);
        // Var(y|h) = exp(h) * nu/(nu-2) for the raw Student-t scale this
        // likelihood uses (see model block); vol_forecast alone understates
        // the SD by sqrt(nu/(nu-2)). use_student_t=0 -> nu is +inf, so the
        // ratio must be branched, not computed as inf/inf.
        sd_forecast = use_student_t ? vol_forecast * sqrt(nu / (nu - 2)) : vol_forecast;
    }
}

// ---------------------------------------------------------------------------
// If h_{t-1} really must enter the network (state-dependent mean reversion),
// the exact centring above is no longer possible, because g_t then depends on
// h_{t-1}.  The least damaging version is:
//
//   * feed the standardised deviation  (h[t-1] - mu) * sqrt(1 - phi^2)/sigma_eta,
//     never h[t-1] itself;
//   * drop b1 AND b2, so that the map z -> s*tanh(w2' tanh(W1 z)) is an odd
//     function; with mean-zero inputs its output is then approximately
//     mean-zero and only weakly confounded with mu;
//   * expect a slower sampler, because h becomes a recurrent nonlinear function
//     of eta_raw.
// Even so, mu will be less well identified than in the version above.
// ---------------------------------------------------------------------------
