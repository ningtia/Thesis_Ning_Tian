data {
  int<lower=2> T;
  int<lower=1> D;
  vector[T] y;
  matrix[T, D] X;
}

parameters {
  ordered[2] mu;
  vector<lower=0, upper=0.999>[2] phi;
  vector<lower=0>[2] sigma_eta;
  vector[D] beta;
  vector[T] h;
  real<lower=1e-6, upper=0.999999> p11;
  real<lower=1e-6, upper=0.999999> p22;
  real<lower=0> nu_minus2;
}

transformed parameters {
  real<lower=2> nu = 2 + nu_minus2;
  matrix[2, 2] P;

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

  mu ~ normal(0, 5);
  phi ~ beta(20, 1.5);
  sigma_eta ~ normal(0, 1);
  beta ~ normal(0, 1);
  p11 ~ beta(20, 2);
  p22 ~ beta(20, 2);
  nu_minus2 ~ gamma(2, 0.1);

  denom = 2 - p11 - p22;
  pi_stat[1] = (1 - p22) / denom;
  pi_stat[2] = (1 - p11) / denom;

  for (s in 1:2) {
    log_alpha[s] = log(pi_stat[s])
                   + normal_lpdf(
                       h[1] |
                       mu[s] + X[1] * beta,
                       sigma_eta[s] / sqrt(1 - square(phi[s]))
                     );
  }

  for (t in 2:T) {
    for (s in 1:2) {
      vector[2] candidates;
      for (r in 1:2) {
        candidates[r] = log_alpha[r] + log(P[r, s]);
      }
      log_alpha_next[s] = log_sum_exp(candidates)
                          + normal_lpdf(
                              h[t] |
                              mu[s]
                              + phi[s] * (h[t - 1] - mu[s])
                              + X[t] * beta,
                              sigma_eta[s]
                            );
    }
    log_alpha = log_alpha_next;
  }

  target += log_sum_exp(log_alpha);

  for (t in 1:T) {
    y[t] ~ student_t(nu, 0, exp(0.5 * h[t]));
  }
}

generated quantities {
  vector[2] filtered_prob_last;
  vector[2] pi_stat;
  vector[2] log_alpha;
  vector[2] log_alpha_next;
  real denom;

  denom = 2 - p11 - p22;
  pi_stat[1] = (1 - p22) / denom;
  pi_stat[2] = (1 - p11) / denom;

  for (s in 1:2) {
    log_alpha[s] = log(pi_stat[s])
                   + normal_lpdf(
                       h[1] |
                       mu[s] + X[1] * beta,
                       sigma_eta[s] / sqrt(1 - square(phi[s]))
                     );
  }
  for (t in 2:T) {
    for (s in 1:2) {
      vector[2] candidates;
      for (r in 1:2) {
        candidates[r] = log_alpha[r] + log(P[r, s]);
      }
      log_alpha_next[s] = log_sum_exp(candidates)
                          + normal_lpdf(
                              h[t] |
                              mu[s]
                              + phi[s] * (h[t - 1] - mu[s])
                              + X[t] * beta,
                              sigma_eta[s]
                            );
    }
    log_alpha = log_alpha_next;
  }
  filtered_prob_last = softmax(log_alpha);
}
