source("benchmark_utils.R")

refit_every <- getOption("benchmark.refit_every", 13L)
har_predictive_draws <- getOption("benchmark.har_predictive_draws", 5000L)
bench <- load_benchmark_data()

lagged_mean <- function(x, window) {
  out <- rep(NA_real_, length(x))
  for (i in seq_along(x)) {
    if (i > window) {
      out[i] <- mean(x[(i - window):(i - 1L)], na.rm = FALSE)
    }
  }
  out
}

# All HAR regressors for row t only use realised variance through t-1.  The
# features may therefore be prepared once, without exposing future returns.
har_data <- bench$all
har_data$.row_index <- seq_len(nrow(har_data))
har_data$.rv <- safe_forecast_variance(
  actual_return_variance(har_data[[bench$target_col]])
)
har_data$.log_rv <- log(har_data$.rv)
har_data$.log_rv_lag_1 <- log(safe_forecast_variance(lagged_mean(har_data$.rv, 1L)))
har_data$.log_rv_lag_4 <- log(safe_forecast_variance(lagged_mean(har_data$.rv, 4L)))
har_data$.log_rv_lag_13 <- log(safe_forecast_variance(lagged_mean(har_data$.rv, 13L)))

har_cols <- c(".log_rv_lag_1", ".log_rv_lag_4", ".log_rv_lag_13")
required_cols <- c(".log_rv", har_cols, bench$covariates)
har_data <- har_data[stats::complete.cases(har_data[, required_cols, drop = FALSE]), , drop = FALSE]
har_formula <- stats::as.formula(
  paste(".log_rv ~", paste(c(har_cols, bench$covariates), collapse = " + "))
)

fit_har_rv <- function(train_indices, previous_model, refit_id, bench) {
  training_data <- har_data[har_data$.row_index %in% train_indices, , drop = FALSE]
  fit <- stats::lm(har_formula, data = training_data)
  list(
    model = list(fit = fit, sigma2_log = stats::sigma(fit)^2),
    state = list(),
    fit = fit,
    diagnostics = list(
      nobs = stats::nobs(fit),
      r_squared = summary(fit)$r.squared
    )
  )
}

forecast_har_one_step <- function(model, state, forecast_index, bench) {
  newdata <- har_data[har_data$.row_index == forecast_index, , drop = FALSE]
  if (nrow(newdata) != 1L) {
    stop(sprintf("HAR data are unavailable for forecast row %d.", forecast_index))
  }
  predicted_log_rv <- as.numeric(stats::predict(model$fit, newdata = newdata))
  variance_t <- safe_forecast_variance(exp(predicted_log_rv + 0.5 * model$sigma2_log))
  sigma_t <- sqrt(variance_t)

  list(
    variance_draws = rep(variance_t, har_predictive_draws),
    # HAR-RV supplies a variance forecast rather than a return innovation
    # distribution.  Tail measures consequently use its conventional
    # conditionally Gaussian zero-mean benchmark distribution.
    return_draws = stats::rnorm(har_predictive_draws, mean = 0, sd = sigma_t),
    state_prior = list(),
    log_density_draws = function(y) {
      rep(stats::dnorm(y, mean = 0, sd = sigma_t, log = TRUE), har_predictive_draws)
    }
  )
}

update_har_state <- function(model, state, forecast, observed_y,
                             observation_index, bench) {
  # The next row's HAR lags were built from realised observations only, so no
  # parameter refit is needed until the next quarterly expanding-window fit.
  list()
}

rolling_har_rv <- rolling_forecast(
  bench = bench,
  fit_model = fit_har_rv,
  forecast_one_step = forecast_har_one_step,
  update_state = update_har_state,
  refit_every = refit_every,
  model_name = "HAR_RV",
  seed = 5666L
)

if (!exists("benchmark_results")) {
  benchmark_results <- list()
}
benchmark_results$HAR_RV <- rolling_har_rv
