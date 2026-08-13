# library(rugarch)

if (!exists("rolling_forecast", mode = "function", inherits = TRUE)) {
  source("benchmark_utils.R")
}

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

garch_one_step_parameters <- function(model, state, forecast_index, bench) {
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
  list(
    sigma_t = as.numeric(rugarch::sigma(forecast))[1L],
    nu_t = unname(model$coefficients["shape"])
  )
}

validate_garch_candidate <- function(fit, model, state, forecast_index, bench) {
  coefficients <- fit@fit$coef
  if (fit@fit$convergence != 0L) {
    return(sprintf("optimizer convergence code %s", fit@fit$convergence))
  }
  if (!length(coefficients) || any(!is.finite(coefficients))) {
    return("non-finite fitted coefficients")
  }

  probe <- tryCatch(
    garch_one_step_parameters(model, state, forecast_index, bench),
    error = function(e) e
  )
  if (inherits(probe, "error")) {
    return(paste("one-step forecast failed:", conditionMessage(probe)))
  }
  if (!is.finite(probe$sigma_t) || probe$sigma_t <= 0) {
    return("one-step forecast returned a non-finite or non-positive sigma")
  }
  if (!is.finite(probe$nu_t) || probe$nu_t <= 2) {
    return("one-step forecast returned an invalid Student-t shape")
  }
  NULL
}

fit_garch_model <- function(model_name,
                            solver = "hybrid",
                            fallback_solver = NULL) {
  function(train_indices, previous_model, refit_id, bench) {
    starting_values <- if (is.null(previous_model)) list() else {
      as.list(previous_model$coefficients)
    }
    solvers <- unique(c(solver, fallback_solver))
    failures <- character()

    for (solver_name in solvers) {
      spec <- make_garch_spec(
        model_name = model_name,
        x = bench$X_all[train_indices, , drop = FALSE],
        start_pars = starting_values
      )
      fit <- tryCatch(
        rugarch::ugarchfit(
          spec = spec,
          data = bench$y_all[train_indices],
          solver = solver_name
        ),
        error = function(e) e
      )
      if (inherits(fit, "error")) {
        failures <- c(failures, paste(solver_name, conditionMessage(fit)))
        next
      }

      model <- list(
        model_name = model_name,
        solver = solver_name,
        coefficients = fit@fit$coef
      )
      state <- list(history_indices = train_indices)
      validation_error <- validate_garch_candidate(
        fit = fit,
        model = model,
        state = state,
        forecast_index = max(train_indices) + 1L,
        bench = bench
      )
      if (!is.null(validation_error)) {
        failures <- c(failures, paste(solver_name, validation_error))
        next
      }

      return(list(
        model = model,
        # Between semiannual re-estimations this index grows by one observation;
        # the fixed-parameter spec is filtered, not re-estimated.
        state = state,
        fit = fit,
        diagnostics = list(
          convergence = fit@fit$convergence,
          persistence = rugarch::persistence(fit),
          solver = solver_name
        )
      ))
    }

    stop(
      sprintf(
        "%s could not produce a valid refit %d: %s",
        model_name, refit_id, paste(failures, collapse = " | ")
      ),
      call. = FALSE
    )
  }
}

forecast_garch_one_step <- function(model, state, forecast_index, bench) {
  forecast_parameters <- garch_one_step_parameters(
    model, state, forecast_index, bench
  )
  sigma_t <- forecast_parameters$sigma_t
  if (!is.finite(sigma_t) || sigma_t <= 0) {
    stop("GARCH forecast returned an invalid conditional standard deviation.",
         call. = FALSE)
  }
  nu_t <- forecast_parameters$nu_t
  if (!is.finite(nu_t) || nu_t <= 2) {
    stop("The Student-t GARCH shape parameter must be greater than 2.",
         call. = FALSE)
  }

  # rugarch's "std" innovations are Student-t innovations standardized to
  # unit variance, so sigma_t remains the conditional standard deviation.
  t_scale <- sigma_t * sqrt((nu_t - 2) / nu_t)
  predictive_draws <- getOption("benchmark.garch_predictive_draws", 5000L)
  return_draws <- stats::rt(predictive_draws, df = nu_t) * t_scale

  list(
    variance_draws = rep(sigma_t^2, predictive_draws),
    return_draws = return_draws,
    state_prior = list(history_indices = state$history_indices),
    log_density_draws = function(y) {
      rep(stats::dt(y / t_scale, df = nu_t, log = TRUE) - log(t_scale),
          predictive_draws)
    }
  )
}

update_garch_state <- function(model, state, forecast, observed_y,
                               observation_index, bench) {
  list(history_indices = c(state$history_indices, observation_index))
}

run_garch_benchmark <- function(model_name,
                                bench = load_benchmark_data(),
                                refit_every = getOption("benchmark.refit_every", 13L),
                                save_fits = getOption("benchmark.save_fits", FALSE),
                                seed = 666L,
                                solver = "hybrid",
                                fallback_solver = NULL) {
  rolling_forecast(
    bench = bench,
    fit_model = fit_garch_model(model_name, solver, fallback_solver),
    forecast_one_step = forecast_garch_one_step,
    update_state = update_garch_state,
    refit_every = refit_every,
    model_name = model_name,
    save_fits = save_fits,
    seed = seed
  )
}

if (isTRUE(getOption("benchmark.garch_autorun", FALSE))) {
  bench <- load_benchmark_data()
  if (!exists("benchmark_results")) benchmark_results <- list()
  benchmark_results$sGARCH <- run_garch_benchmark("sGARCH", bench = bench)
  benchmark_results$gjrGARCH <- run_garch_benchmark("gjrGARCH", bench = bench)
  benchmark_results$eGARCH <- run_garch_benchmark(
    "eGARCH", bench = bench, solver = "hybrid", fallback_solver = "solnp"
  )
}
