data {
  int<lower=2> T;
  int<lower=1> D;
  vector[T] y;
  matrix[T, D] X;

  int<lower=0, upper=1> stationary_init;   // 1 = stationary h[1,s] (explodes as phi -> 1)
  real<lower=0> h1_scale;                  // scale of h[1,s] when stationary_init = 0

  // 2026-08-25: mu/phi/nu priors now match nonlinear_sv.stan's (same named
  // hyperparameters, passed as data instead of hardcoded, same default
  // values: mu_scale=5, phi_a=20, phi_b=1.5, nu_rate=0.1) so the three
  // parameters this model shares with the rest of the SV family use one
  // consistent prior across all of them. sigma_eta is deliberately NOT
  // included here -- see the model block below.
  real mu_scale;                           // sd of the normal prior on each mu[s]
  real<lower=0> phi_a;                     // beta shape a for phi_unit
  real<lower=0> phi_b;                     // beta shape b for phi_unit
  real<lower=0> nu_rate;                   // gamma rate for nu - 2 (shape 2)
}

parameters {
  ordered[2] mu;
  real<lower=0, upper=1> phi_unit;
  real log_sigma_eta;
  vector[D] beta;
  matrix[T, 2] eta_raw;
  real<lower=0, upper=1> p11_unit;
  real<lower=0, upper=1> p22_unit;
  real<lower=0> nu_minus2;
}

transformed parameters {
  real<lower=2> nu = 2 + nu_minus2;
  real<lower=0, upper=0.995> phi;
  real<lower=0> sigma_eta;
  real<lower=0.005, upper=0.995> p11;
  real<lower=0.005, upper=0.995> p22;
  matrix[T, 2] h;
  matrix[2, 2] P;

  phi = 0.995 * phi_unit;
  sigma_eta = exp(log_sigma_eta);
  p11 = 0.005 + 0.99 * p11_unit;
  p22 = 0.005 + 0.99 * p22_unit;

  // Each regime has an independent non-centered log-volatility path.
  // Conditional on regime s, this is h_t,s = mu_s + phi(h_t-1,s - mu_s)
  // + X_t beta + sigma_eta epsilon_t,s, with epsilon_t,s ~ N(0, 1).
  //
  // The stationary scale sigma_eta / sqrt(1 - phi^2) diverges as phi -> 1 and
  // couples phi, sigma_eta and eta_raw[1,s] into a funnel -- same failure
  // mode as nonlinear_sv.stan before its 2026-08-21 fix, and phi_unit's
  // beta(phi_a, phi_b) prior below (default 20, 1.5) pushes phi toward its
  // 0.995 ceiling the same way. With stationary_init = 0 each regime's initial state gets a fixed
  // diffuse scale instead; one observation carries almost no likelihood
  // weight, so this costs nothing.
  for (s in 1:2) {
    h[1, s] = mu[s] + X[1] * beta
              + (stationary_init
                 ? sigma_eta / sqrt(1 - square(phi))
                 : h1_scale) * eta_raw[1, s];
  }
  for (t in 2:T) {
    for (s in 1:2) {
      h[t, s] = mu[s] + phi * (h[t - 1, s] - mu[s])
                + X[t] * beta + sigma_eta * eta_raw[t, s];
    }
  }

  P[1, 1] = p11;
  P[1, 2] = 1 - p11;
  P[2, 1] = 1 - p22;
  P[2, 2] = p22;
}

model {
  vector[2] log_alpha;
  vector[2] log_alpha_next;
  vector[2] pi_stat;
  real denom;
  real log_normalizer;

  mu ~ normal(0, mu_scale);
  phi_unit ~ beta(phi_a, phi_b);
  // log_sigma_eta keeps its own unconstrained, log-normal-shaped prior --
  // NOT sigma_eta ~ normal(0, sigma_eta_scale) like the rest of the SV
  // family. Deliberate: see docs/superpowers/specs/2026-08-15-ms-sv-
  // stabilization-design.md ("Do not copy linear-SV non-centering directly:
  // MS-SV marginalizes a discrete mixture and requires a different
  // parameterization"). Copying the half-normal-on-the-natural-scale prior
  // here would let sigma_eta sit arbitrarily close to 0 again, reopening the
  // funnel this parameterization was chosen to avoid.
  log_sigma_eta ~ normal(log(0.25), 0.75);
  beta ~ normal(0, 1);
  p11_unit ~ beta(20, 1.5);
  p22_unit ~ beta(20, 1.5);
  nu_minus2 ~ gamma(2, nu_rate);
  to_vector(eta_raw) ~ std_normal();

  denom = 2 - p11 - p22;
  pi_stat[1] = (1 - p22) / denom;
  pi_stat[2] = (1 - p11) / denom;

  // Scaled forward algorithm: the discrete regime is marginalized, while
  // the regime-specific non-centered volatility paths remain continuous.
  for (s in 1:2) {
    log_alpha[s] = log(pi_stat[s])
                   + student_t_lpdf(y[1] | nu, 0, exp(0.5 * h[1, s]));
  }
  log_normalizer = log_sum_exp(log_alpha);
  target += log_normalizer;
  log_alpha -= log_normalizer;

  for (t in 2:T) {
    for (s in 1:2) {
      vector[2] candidates;
      for (r in 1:2) {
        candidates[r] = log_alpha[r] + log(P[r, s]);
      }
      log_alpha_next[s] = log_sum_exp(candidates)
                          + student_t_lpdf(
                              y[t] | nu, 0, exp(0.5 * h[t, s])
                            );
    }
    log_normalizer = log_sum_exp(log_alpha_next);
    target += log_normalizer;
    log_alpha = log_alpha_next - log_normalizer;
  }
}

generated quantities {
  vector[T] log_lik;
  vector[2] filtered_prob_last;
  vector[2] pi_stat;
  vector[2] log_alpha;
  vector[2] log_alpha_next;
  real denom;
  real log_normalizer;

  denom = 2 - p11 - p22;
  pi_stat[1] = (1 - p22) / denom;
  pi_stat[2] = (1 - p11) / denom;

  for (s in 1:2) {
    log_alpha[s] = log(pi_stat[s])
                   + student_t_lpdf(y[1] | nu, 0, exp(0.5 * h[1, s]));
  }
  log_normalizer = log_sum_exp(log_alpha);
  log_lik[1] = log_normalizer;
  log_alpha -= log_normalizer;

  for (t in 2:T) {
    for (s in 1:2) {
      vector[2] candidates;
      for (r in 1:2) {
        candidates[r] = log_alpha[r] + log(P[r, s]);
      }
      log_alpha_next[s] = log_sum_exp(candidates)
                          + student_t_lpdf(
                              y[t] | nu, 0, exp(0.5 * h[t, s])
                            );
    }
    log_normalizer = log_sum_exp(log_alpha_next);
    log_lik[t] = log_normalizer;
    log_alpha = log_alpha_next - log_normalizer;
  }

  filtered_prob_last = softmax(log_alpha);
}
