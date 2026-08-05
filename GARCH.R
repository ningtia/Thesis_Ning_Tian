library(rugarch)

source("benchmark_utils.R")

refit_every <- getOption("benchmark.refit_every", 13L)
garch_predictive_draws <- getOption("benchmark.garch_predictive_draws", 5000L)
bench <- load_benchmark_data()

make_garch_spec <- function(model_name, x, fixed_pars = list(), start_pars = list()) {
  rugarch::ugarchspec(
    variance.model = list(
      model = model_name,
      garchOrder = c(1, 1),
      external.regressors = as.matrix(x)
    ),
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
    distribution.model = "std",
    fixed.pars = fixed_pars,
    start.pars = start_pars
  )
}

fit_garch_model <- function(model_name, solver = "hybrid") {
  function(train_indices, previous_model, refit_id, bench) {
    starting_values <- if (is.null(previous_model)) list() else {
      as.list(previous_model$coefficients)
    }
    spec <- make_garch_spec(
      model_name = model_name,
      x = bench$X_all[train_indices, , drop = FALSE],
      start_pars = starting_values
    )
    fit <- rugarch::ugarchfit(
      spec = spec,
      data = bench$y_all[train_indices],
      solver = solver
    )
    if (fit@fit$convergence != 0L) {
      warning(sprintf("%s did not converge at refit %d.", model_name, refit_id))
    }

    list(
      model = list(
        model_name = model_name,
        solver = solver,
        coefficients = fit@fit$coef
      ),
      # Between quarterly re-estimations this index grows by one observation;
      # the fixed-parameter spec is filtered, not re-estimated.
      state = list(history_indices = train_indices),
      fit = fit,
      diagnostics = list(
        convergence = fit@fit$convergence,
        persistence = rugarch::persistence(fit)
      )
    )
  }
}

forecast_garch_one_step <- function(model, state, forecast_index, bench) {
  history_indices <- state$history_indices
  fixed_spec <- make_garch_spec(
    model_name = model$model_name,
    x = bench$X_all[history_indices, , drop = FALSE],
    fixed_pars = as.list(model$coefficients)
  )
  forecast <- rugarch::ugarchforecast(
    fitORspec = fixed_spec,
    data = bench$y_all[history_indices],
    n.ahead = 1L,
    external.forecasts = list(
      vregfor = matrix(bench$X_all[forecast_index, ], nrow = 1L)
    )
  )
  sigma_t <- as.numeric(sigma(forecast))[1L]
  if (!is.finite(sigma_t) || sigma_t <= 0) {
    warning("GARCH forecast returned an invalid conditional standard deviation.")
  }
  nu_t <- unname(model$coefficients["shape"])
  if (!is.finite(nu_t) || nu_t <= 2) {
    warning("The Student-t GARCH shape parameter must be greater than 2.")
  }

  # rugarch's "std" innovations are Student-t innovations standardized to
  # unit variance, so sigma_t remains the conditional standard deviation.
  t_scale <- sigma_t * sqrt((nu_t - 2) / nu_t)
  return_draws <- stats::rt(garch_predictive_draws, df = nu_t) * t_scale

  list(
    variance_draws = rep(sigma_t^2, garch_predictive_draws),
    return_draws = return_draws,
    state_prior = list(history_indices = history_indices),
    log_density_draws = function(y) {
      rep(stats::dt(y / t_scale, df = nu_t, log = TRUE) - log(t_scale),
          garch_predictive_draws)
    }
  )
}

update_garch_state <- function(model, state, forecast, observed_y,
                               observation_index, bench) {
  list(history_indices = c(state$history_indices, observation_index))
}

run_garch_benchmark <- function(model_name, solver = "hybrid") {
  rolling_forecast(
    bench = bench,
    fit_model = fit_garch_model(model_name, solver),
    forecast_one_step = forecast_garch_one_step,
    update_state = update_garch_state,
    refit_every = refit_every,
    model_name = model_name,
    seed = switch(model_name, sGARCH = 2666L, gjrGARCH = 3666L, eGARCH = 4666L)
  )
}

if (!exists("benchmark_results")) {
  benchmark_results <- list()
}
benchmark_results$sGARCH <- run_garch_benchmark("sGARCH")
benchmark_results$gjrGARCH <- run_garch_benchmark("gjrGARCH")
benchmark_results$eGARCH <- run_garch_benchmark("eGARCH", solver = "gosolnp")
