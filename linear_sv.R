library(rstan)

source("benchmark_utils.R")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

refit_every <- getOption("benchmark.refit_every", 13)
save_rolling_fits <- getOption("benchmark.save_fits", FALSE)
stan_iter <- getOption("benchmark.stan_iter", 4000)
stan_warmup <- getOption("benchmark.stan_warmup", 2000)
stan_chains <- getOption("benchmark.stan_chains", 4)

bench <- load_benchmark_data()

linear_sv_stan_data <- function(bench, train_indices) {
  y_train <- bench$y_all[train_indices]

  list(
    T = length(train_indices),
    K = ncol(bench$X_all),
    y = y_train,
    X = bench$X_all[train_indices, , drop = FALSE],
    h0_location = log(mean(actual_return_variance(y_train), na.rm = TRUE))
  )
}

linear_sv_init <- function(stan_data, previous_post = NULL) {
  if (is.null(previous_post)) {
    return(list(
      mu = stan_data$h0_location,
      phi_raw = atanh(0.95),
      sigma_eta = 0.25,
      beta = rep(0, stan_data$K),
      eta_raw = rep(0, stan_data$T),
      nu_minus2 = 8
    ))
  }

  list(
    mu = stats::median(previous_post$mu),
    phi_raw = stats::median(previous_post$phi_raw),
    sigma_eta = max(stats::median(previous_post$sigma_eta), 1e-4),
    beta = apply(previous_post$beta, 2, stats::median),
    eta_raw = rep(0, stan_data$T),
    nu_minus2 = max(stats::median(previous_post$nu_minus2), 1e-4)
  )
}

forecast_linear_sv_chunk <- function(post, bench, chunk) {
  forecast_indices <- chunk$forecast_indices
  X_out <- bench$X_all[forecast_indices, , drop = FALSE]
  n_out <- nrow(X_out)
  n_draws <- length(post$mu)

  var_draws <- matrix(NA_real_, n_draws, n_out)
  h_prev <- post$h[, length(chunk$train_indices)]
  nu_multiplier <- post$nu / (post$nu - 2)

  for (t in seq_len(n_out)) {
    xb <- as.vector(post$beta %*% X_out[t, ])
    h_now <- post$mu + post$phi * (h_prev - post$mu) + xb
    var_draws[, t] <- exp(h_now + 0.5 * post$sigma_eta^2) * nu_multiplier
    h_prev <- h_now
  }

  safe_forecast_variance(colMeans(var_draws))
}

###########################
# 13-week expanding-window re-estimation.
###########################

chunks <- make_oos_chunks(
  bench,
  refit_every = refit_every
)
oos_indices <- sort(c(bench$split_rows$valid, bench$split_rows$test))

forecast_variance <- rep(NA_real_, length(oos_indices))
rolling_diagnostics <- vector("list", length(chunks))
rolling_fits <- if (isTRUE(save_rolling_fits)) {
  vector("list", length(chunks))
} else {
  NULL
}

previous_post <- NULL
last_fit <- NULL
last_post <- NULL

for (i in seq_along(chunks)) {
  chunk <- chunks[[i]]
  stan_train <- linear_sv_stan_data(bench, chunk$train_indices)

  fit_linear <- stan(
    file = "linear_sv.stan",
    data = stan_train,
    chains = stan_chains,
    iter = stan_iter,
    warmup = stan_warmup,
    seed = 666 + i - 1,
    init = function() linear_sv_init(stan_train, previous_post),
    control = list(adapt_delta = 0.95, max_treedepth = 12)
  )

  post_linear <- rstan::extract(fit_linear)
  forecast_variance[chunk$oos_positions] <- forecast_linear_sv_chunk(
    post_linear,
    bench,
    chunk
  )

  rolling_diagnostics[[i]] <- list(
    chunk_id = chunk$chunk_id,
    train_T = length(chunk$train_indices),
    forecast_start = chunk$start_date,
    forecast_end = chunk$end_date,
    stan = stan_fit_diagnostics(fit_linear)
  )

  if (isTRUE(save_rolling_fits)) {
    rolling_fits[[i]] <- fit_linear
  }

  previous_post <- post_linear
  last_fit <- fit_linear
  last_post <- post_linear
}

forecast_split <- split_oos_forecasts(
  bench,
  oos_indices,
  forecast_variance
)

var_valid <- forecast_split$var_valid
var_test <- forecast_split$var_test
vol_valid <- forecast_split$vol_valid
vol_test <- forecast_split$vol_test

qlike_valid <- qlike(forecast_split$actual_valid, var_valid)
qlike_test <- qlike(forecast_split$actual_test, var_test)

if (!exists("benchmark_results")) {
  benchmark_results <- list()
}

benchmark_results$linearSV <- list(
  fit = last_fit,
  posterior = last_post,
  rolling_fits = rolling_fits,
  forecasts = list(
    forecast_valid = var_valid,
    forecast_test = var_test,
    var_valid = var_valid,
    var_test = var_test,
    vol_valid = vol_valid,
    vol_test = vol_test,
    qlike_valid = qlike_valid,
    qlike_test = qlike_test,
    oos_indices = oos_indices,
    refit_every = refit_every
  ),
  diagnostics = list(
    rolling = rolling_diagnostics
  ),
  data = list(
    target_col = bench$target_col,
    covariates = bench$covariates,
    scaling = bench$scaling
  )
)
