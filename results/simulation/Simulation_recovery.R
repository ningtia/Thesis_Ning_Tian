# Simulation_recovery.R
#
# Scope narrowed 2026-08-21: this now checks ONLY the one thing no other
# script in this folder checks -- does the fitted latent path (h_t, the
# log-variance; g_t, the nonlinear term) actually track the TRUE simulated
# path over time, with per-time-point 95% coverage and RMSE? That is a
# materially different question from "is the scalar summary (mu, g_sd,
# h_bar, ...) close to its true value", and matters for a thesis about
# volatility forecasting: a model can get g_sd/h_bar right on average while
# still tracking the wrong shape.
#
# Everything else this script used to do -- the mu/phi/sigma_eta/nu/g_sd/
# h_bar recovery table, MCMC/HMC diagnostics -- is now done, more
# thoroughly (true_value/bias/inside_95 columns, per-chain diagnostics,
# prior-vs-posterior comparison), by prior_predictive_check.R's "main"
# scenario, which fits this exact same nnsv_simulation_main.csv. Re-running
# both was pure duplication; if you want that table, it's
# nnsv_prior_vs_posterior_main.csv now.
#
# A single fit here is still a single draw (the "coin flip" recovery_
# replications.R's header warns about) -- treat the coverage numbers below
# as illustrative, not as a formal claim, unless corroborated by repetition.

library(rstan)
library(posterior)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Reuse the same warm-start init as prior_predictive_check.R (data-driven
# mu/phi/sigma_eta, NN weights near zero) instead of an undirected init_r
# spread. See that script for why the autorun option and chdir matter here.
options(benchmark.nn_autorun = FALSE)
source("../../nonlinear_sv.R", chdir = TRUE)

sim_data <- read.csv("nnsv_simulation_main.csv")
T_sim <- nrow(sim_data)

K_hidden <- 4

stan_data <- list(
  T = T_sim,
  D = 1,
  K = K_hidden,
  y = sim_data$y,
  X = matrix(sim_data$x[1:(T_sim - 1)], ncol = 1),
  use_student_t = 1,
  x_forecast = array(sim_data$x[T_sim], dim = 1),

  s_fixed     = 0.50,
  tau_w_scale = 0.50,

  # prior hyperparameters (vary these for the sensitivity table)
  mu_scale        = 5,
  phi_a           = 20,
  phi_b           = 1.5,
  sigma_eta_scale = 1,
  b1_scale        = 0.5,
  nu_rate         = 0.1,

  prior_only = 0L,        # 0 = 正常后验估计;1 = 仅先验预测检查
  stationary_init = 0L,   # 使用 v3 的固定尺度初始状态
  h1_scale = 2.0          # h_1 相对 mu 的固定扩散尺度
)

model <- stan_model("nonlinear_sv_v3.stan")

linear_start <- nn_sv_linear_start(stan_data$y, stan_data$X)
init_fn <- function() nn_sv_init(stan_data, previous_post = NULL, linear_start = linear_start)

fit <- sampling(
  model,
  data    = stan_data,
  chains  = 4,
  cores   = 4,
  iter    = 4000,
  warmup  = 1500,
  seed    = 20260821,
  refresh = 250,
  init    = init_fn,
  control = list(adapt_delta = 0.999, max_treedepth = 12)
)

saveRDS(fit, "nnsv_simulation_fit.rds")

# ---------------------------------------------------------------------------
# Minimal convergence gate. This is NOT a substitute for the full
# diagnostics -- those live in nnsv_prior_predictive_hmc_main.csv /
# nnsv_prior_predictive_mcmc_main.csv (prior_predictive_check.R) -- just
# enough to know whether the path-recovery numbers below are worth reading.
# ---------------------------------------------------------------------------
sampler_parameters <- get_sampler_params(fit, inc_warmup = FALSE)
ndiv <- sum(sapply(sampler_parameters, function(p) sum(p[, "divergent__"])))
all_diag <- posterior::summarise_draws(posterior::as_draws_array(fit))
converged <- ndiv == 0 &&
  max(all_diag$rhat, na.rm = TRUE) < 1.01 &&
  min(all_diag$ess_bulk, na.rm = TRUE) > 400
message("Usable fit: ", converged, " (divergences = ", ndiv, ")")

# ---------------------------------------------------------------------------
# Latent path h and g, taken straight from Stan (no re-implementation in R).
#    The fitted g is mean-zero by construction, so the true g is centred too;
#    only the shape of the function is identified separately from mu.
# ---------------------------------------------------------------------------
draws <- rstan::extract(fit, pars = c("g", "h"))

h_median <- apply(draws$h, 2, median)
h_lower  <- apply(draws$h, 2, quantile, probs = 0.025)
h_upper  <- apply(draws$h, 2, quantile, probs = 0.975)

h_recovery <- data.frame(
  time = sim_data$time, true_h = sim_data$h_true,
  posterior_median = h_median, lower_95 = h_lower, upper_95 = h_upper
)
h_summary <- data.frame(
  quantity    = "h",
  rmse        = sqrt(mean((h_median - sim_data$h_true)^2)),
  coverage_95 = mean(sim_data$h_true >= h_lower & sim_data$h_true <= h_upper)
)

g_draws  <- draws$g                                   # iterations x (T-1)
true_g_c <- sim_data$g_true[2:T_sim] - mean(sim_data$g_true[2:T_sim])

g_median <- apply(g_draws, 2, median)
g_lower  <- apply(g_draws, 2, quantile, probs = 0.025)
g_upper  <- apply(g_draws, 2, quantile, probs = 0.975)

g_recovery <- data.frame(
  time = sim_data$time[2:T_sim], true_g_centred = true_g_c,
  posterior_median = g_median, lower_95 = g_lower, upper_95 = g_upper
)
g_summary <- data.frame(
  quantity    = "g (centred)",
  rmse        = sqrt(mean((g_median - true_g_c)^2)),
  coverage_95 = mean(true_g_c >= g_lower & true_g_c <= g_upper)
)

state_summary <- rbind(h_summary, g_summary)
write.csv(h_recovery, "nnsv_h_recovery.csv", row.names = FALSE)
write.csv(g_recovery, "nnsv_g_recovery.csv", row.names = FALSE)
write.csv(state_summary,"nnsv_state_recovery_summary.csv", row.names = FALSE)
print(state_summary)
