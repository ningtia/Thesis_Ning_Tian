data {
  int<lower=2> T;
  int<lower=1> K;
  vector[T] y;
  matrix[T, K] X;
  real h0_location;
}

parameters {
  real mu;
  real phi_raw;
  real<lower=0> sigma_eta;
  vector[K] beta;
  vector[T] eta_raw;
  real<lower=0> nu_minus2;
}

transformed parameters {
  real phi = tanh(phi_raw);
  real<lower=2> nu = 2 + nu_minus2;
  vector[T] h;

  h[1] = mu + sigma_eta / sqrt(1 - square(phi)) * eta_raw[1];

  for (t in 2:T) {
    h[t] = mu
           + phi * (h[t - 1] - mu)
           + X[t] * beta
           + sigma_eta * eta_raw[t];
  }
}

model {
  mu ~ normal(h0_location, 2);
  phi_raw ~ normal(2, 1);
  sigma_eta ~ normal(0, 1);
  beta ~ normal(0, 1);
  eta_raw ~ normal(0, 1);
  nu_minus2 ~ gamma(2, 0.1);

  for (t in 1:T) {
    y[t] ~ student_t(nu, 0, exp(0.5 * h[t]));
  }
}

generated quantities {
  vector[T] log_lik;

  for (t in 1:T) {
    log_lik[t] = student_t_lpdf(y[t] | nu, 0, exp(0.5 * h[t]));
  }
}
