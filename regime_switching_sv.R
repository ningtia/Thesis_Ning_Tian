library(rstan)

source("benchmark_utils.R")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

bench <- load_benchmark_data(
  standardize = TRUE,
  covariates_are_lagged = TRUE
)

h0_location <- log(mean(actual_return_variance(bench$y_train), na.rm = TRUE))

stan_train <- list(
  T = nrow(bench$X_train),
  K = ncol(bench$X_train),
  y = bench$y_train,
  X = bench$X_train,
  h0_location = h0_location
)

init_rs_sv <- function() {
  list(
    mu = c(h0_location - 0.5, h0_location + 0.5),
    phi = c(0.90, 0.95),
    sigma_eta = c(0.25, 0.45),
    beta = rep(0, ncol(bench$X_train)),
    h = rep(h0_location, nrow(bench$X_train)),
    p11 = 0.95,
    p22 = 0.95,
    nu_minus2 = 8
  )
}

###########################
# Model 5: Regime-switching SV (So et al., 1998)
###########################

fit_rs_sv <- stan(
  file = "regime_switching_sv.stan",
  data = stan_train,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  seed = 666,
  init = init_rs_sv,
  control = list(adapt_delta = 0.97, max_treedepth = 13)
)

post_rs_sv <- rstan::extract(fit_rs_sv)

############################################################
## Validation + Test
############################################################

X_out <- rbind(bench$X_valid, bench$X_test)
N_out <- nrow(X_out)
n_valid <- nrow(bench$X_valid)
n_test <- nrow(bench$X_test)
n_draws <- length(post_rs_sv$p11)

var_draws <- matrix(NA_real_, n_draws, N_out)
h_prev <- post_rs_sv$h[, nrow(bench$X_train)]
state_prev <- post_rs_sv$filtered_prob[, nrow(bench$X_train), ]
nu_multiplier <- post_rs_sv$nu / (post_rs_sv$nu - 2)

for (t in seq_len(N_out)) {
  p1_pred <- state_prev[, 1] * post_rs_sv$p11 +
    state_prev[, 2] * (1 - post_rs_sv$p22)
  p2_pred <- state_prev[, 1] * (1 - post_rs_sv$p11) +
    state_prev[, 2] * post_rs_sv$p22

  xb <- as.vector(post_rs_sv$beta %*% X_out[t, ])

  h_mean_1 <- post_rs_sv$mu[, 1] +
    post_rs_sv$phi[, 1] * (h_prev - post_rs_sv$mu[, 1]) +
    xb
  h_mean_2 <- post_rs_sv$mu[, 2] +
    post_rs_sv$phi[, 2] * (h_prev - post_rs_sv$mu[, 2]) +
    xb

  var_draws[, t] <- (
    p1_pred * exp(h_mean_1 + 0.5 * post_rs_sv$sigma_eta[, 1]^2) +
      p2_pred * exp(h_mean_2 + 0.5 * post_rs_sv$sigma_eta[, 2]^2)
  ) * nu_multiplier

  h_prev <- p1_pred * h_mean_1 + p2_pred * h_mean_2
  state_prev <- cbind(p1_pred, p2_pred)
}

forecast_variance <- safe_forecast_variance(colMeans(var_draws))
forecast_volatility <- sqrt(forecast_variance)

var_valid <- forecast_variance[seq_len(n_valid)]
var_test <- forecast_variance[n_valid + seq_len(n_test)]
vol_valid <- forecast_volatility[seq_len(n_valid)]
vol_test <- forecast_volatility[n_valid + seq_len(n_test)]

actual_valid <- actual_return_variance(bench$y_valid)
actual_test <- actual_return_variance(bench$y_test)

qlike_valid <- qlike(actual_valid, var_valid)
qlike_test <- qlike(actual_test, var_test)

summary_fit <- summary(fit_rs_sv)$summary
rhat <- summary_fit[, "Rhat"]
bulk_ess <- summary_fit[, "n_eff"]

sampler_params <- get_sampler_params(fit_rs_sv, inc_warmup = FALSE)
divergence_sum <- sum(sapply(
  sampler_params,
  function(x) sum(x[, "divergent__"])
))

if (!exists("benchmark_results")) {
  benchmark_results <- list()
}

benchmark_results$regimeSwitchingSV <- list(
  fit = fit_rs_sv,
  posterior = post_rs_sv,
  forecasts = list(
    forecast_valid = var_valid,
    forecast_test = var_test,
    var_valid = var_valid,
    var_test = var_test,
    vol_valid = vol_valid,
    vol_test = vol_test,
    qlike_valid = qlike_valid,
    qlike_test = qlike_test
  ),
  diagnostics = list(
    max_rhat = max(rhat, na.rm = TRUE),
    min_bulk_ess = min(bulk_ess, na.rm = TRUE),
    rhat = rhat,
    bulk_ess = bulk_ess,
    divergence_sum = divergence_sum
  ),
  data = list(
    target_col = bench$target_col,
    covariates = bench$covariates,
    covariates_are_lagged = bench$covariates_are_lagged,
    scaling = bench$scaling
  )
)
