if (!exists("rolling_forecast", mode = "function", inherits = TRUE)) {
  source("benchmark_utils.R")
}

lagged_mean <- function(x, window) {
  out <- rep(NA_real_, length(x))
  for (i in seq_along(x)) {
    if (i > window) {
      out[i] <- mean(x[(i - window):(i - 1L)], na.rm = FALSE)
    }
  }
  out
}

read_har_rv_weekly <- function(weekly_file) {
  if (!file.exists(weekly_file)) {
    stop("HAR-RV weekly input file was not found: ", weekly_file, call. = FALSE)
  }

  weekly <- utils::read.csv(weekly_file, stringsAsFactors = FALSE)
  required <- c("Date", "realized_variance_daily_proxy")
  missing_columns <- setdiff(required, names(weekly))
  if (length(missing_columns)) {
    stop(
      "HAR-RV weekly input is missing required column(s): ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  weekly$Date <- as.Date(weekly$Date)
  weekly$.rv <- suppressWarnings(as.numeric(weekly$realized_variance_daily_proxy))
  if (anyNA(weekly$Date) || anyDuplicated(weekly$Date)) {
    stop("HAR-RV weekly input must contain unique, valid dates.", call. = FALSE)
  }
  if (any(!is.finite(weekly$.rv)) || any(weekly$.rv <= 0)) {
    stop("HAR-RV realized-variance proxies must be finite and positive.",
         call. = FALSE)
  }

  weekly[, c("Date", ".rv"), drop = FALSE]
}

prepare_har_rv_data <- function(bench,
                                weekly_file = getOption(
                                  "benchmark.har_rv_weekly_file",
                                  "har_rv_weekly_dataset.csv"
                                )) {
  weekly_rv <- read_har_rv_weekly(weekly_file)
  har_data <- bench$all
  har_data$.row_index <- seq_len(nrow(har_data))
  matched_rows <- match(as.Date(har_data$Date), weekly_rv$Date)
  if (anyNA(matched_rows)) {
    missing_dates <- as.character(har_data$Date[is.na(matched_rows)])
    stop(
      "HAR-RV weekly input is missing realised-variance rows for benchmark date(s): ",
      paste(missing_dates, collapse = ", "),
      call. = FALSE
    )
  }
  har_data$.rv <- weekly_rv$.rv[matched_rows]
  har_data$.log_rv <- log(har_data$.rv)
  har_data$.log_rv_lag_1 <- log(safe_forecast_variance(lagged_mean(har_data$.rv, 1L)))
  har_data$.log_rv_lag_4 <- log(safe_forecast_variance(lagged_mean(har_data$.rv, 4L)))
  har_data$.log_rv_lag_13 <- log(safe_forecast_variance(lagged_mean(har_data$.rv, 13L)))

  har_cols <- c(".log_rv_lag_1", ".log_rv_lag_4", ".log_rv_lag_13")
  required_cols <- c(".log_rv", har_cols, bench$covariates)
  har_data <- har_data[
    stats::complete.cases(har_data[, required_cols, drop = FALSE]),
    ,
    drop = FALSE
  ]
  predictors <- paste(c(har_cols, bench$covariates), collapse = " + ")
  har_formula <- stats::as.formula(paste(".log_rv ~", predictors))

  list(data = har_data, formula = har_formula)
}

# 2026-08-25: HAR-RV used Gaussian return innovations while every other model
# in the benchmark (all three GARCH variants and all three SV models) used
# Student-t. That asymmetry -- not the HAR variance model -- is what failed
# HAR-RV's 1% VaR backtest (5 exceedances against 1.53 expected, p_UC = 0.026)
# while every Student-t model passed. The shape is estimated by MLE on the
# standardised training residuals y_t / sqrt(RVhat_t) rather than fixed, so it
# is a fitted parameter here exactly as rugarch's "shape" is for GARCH.
#
# Set benchmark.har_rv_shape to pin nu instead (e.g. for a robustness check);
# NA, the default, estimates it. har_rv_student_t_shape() returns nu on the
# STANDARDISED-t convention: Var(y) = variance_t regardless of nu, matching
# rugarch's "std" and make_har_rv_forecaster() below.
har_rv_student_t_shape <- function(z,
                                   lower = 2.1,
                                   upper = 100,
                                   default = upper) {
  z <- z[is.finite(z)]
  if (length(z) < 20L) return(default)

  negative_log_likelihood <- function(nu) {
    scale <- sqrt((nu - 2) / nu)
    value <- -sum(stats::dt(z / scale, df = nu, log = TRUE) - log(scale))
    if (!is.finite(value)) return(.Machine$double.xmax)
    value
  }

  optimum <- tryCatch(
    stats::optimize(negative_log_likelihood, interval = c(lower, upper)),
    error = function(e) NULL
  )
  if (is.null(optimum) || !is.finite(optimum$minimum) || optimum$minimum <= 2) {
    return(default)
  }
  optimum$minimum
}

make_har_rv_fitter <- function(har_data, har_formula) {
  force(har_data)
  force(har_formula)

  function(train_indices, previous_model, refit_id, bench) {
    training_data <- har_data[har_data$.row_index %in% train_indices, , drop = FALSE]
    fit <- stats::lm(har_formula, data = training_data)
    sigma2_log <- stats::sigma(fit)^2

    # Same lognormal correction the forecaster applies, so the standardised
    # residuals below are on the scale the predictive distribution actually
    # uses. Rows are complete by construction (prepare_har_rv_data() drops
    # incomplete cases), so training_data and the fitted values line up.
    fitted_variance <- safe_forecast_variance(
      exp(as.numeric(stats::predict(fit, newdata = training_data)) + 0.5 * sigma2_log)
    )
    standardised <- bench$y_all[training_data$.row_index] / sqrt(fitted_variance)

    fixed_shape <- getOption("benchmark.har_rv_shape", NA_real_)
    nu <- if (is.finite(fixed_shape) && fixed_shape > 2) {
      as.numeric(fixed_shape)
    } else {
      har_rv_student_t_shape(standardised)
    }

    list(
      model = list(fit = fit, sigma2_log = sigma2_log, nu = nu),
      state = list(),
      fit = fit,
      diagnostics = list(
        nobs = stats::nobs(fit),
        r_squared = summary(fit)$r.squared,
        shape = nu,
        shape_is_fixed = is.finite(fixed_shape) && fixed_shape > 2
      )
    )
  }
}

make_har_rv_forecaster <- function(har_data, predictive_draws) {
  force(har_data)
  force(predictive_draws)

  function(model, state, forecast_index, bench) {
    newdata <- har_data[har_data$.row_index == forecast_index, , drop = FALSE]
    if (nrow(newdata) != 1L) {
      stop(sprintf("HAR data are unavailable for forecast row %d.", forecast_index))
    }
    predicted_log_rv <- as.numeric(stats::predict(model$fit, newdata = newdata))
    variance_t <- safe_forecast_variance(
      exp(predicted_log_rv + 0.5 * model$sigma2_log)
    )
    sigma_t <- sqrt(variance_t)

    nu_t <- model$nu
    if (is.null(nu_t) || !is.finite(nu_t) || nu_t <= 2) {
      stop("The HAR-RV Student-t shape parameter must be greater than 2.",
           call. = FALSE)
    }

    # Standardised Student-t, the same convention as rugarch's "std" in
    # GARCH.R: the t is scaled so Var(y) = variance_t for any nu, which keeps
    # variance_draws (and therefore QLIKE) unchanged from the Gaussian version
    # and moves only the tail shape -- exactly what the VaR/ES backtests read.
    t_scale <- sigma_t * sqrt((nu_t - 2) / nu_t)

    list(
      variance_draws = rep(variance_t, predictive_draws),
      return_draws = stats::rt(predictive_draws, df = nu_t) * t_scale,
      state_prior = list(),
      log_density_draws = function(y) {
        rep(stats::dt(y / t_scale, df = nu_t, log = TRUE) - log(t_scale),
            predictive_draws)
      }
    )
  }
}

update_har_state <- function(model, state, forecast, observed_y,
                             observation_index, bench) {
  list()
}

run_har_rv <- function(bench = load_benchmark_data(),
                        refit_every = getOption("benchmark.refit_every", 13L),
                        save_fits = getOption("benchmark.save_fits", FALSE),
                        seed = 5666L,
                        weekly_file = getOption(
                          "benchmark.har_rv_weekly_file",
                          "har_rv_weekly_dataset.csv"
                        )) {
  prepared <- prepare_har_rv_data(bench, weekly_file)
  predictive_draws <- getOption("benchmark.har_predictive_draws", 5000L)

  rolling_forecast(
    bench = bench,
    fit_model = make_har_rv_fitter(prepared$data, prepared$formula),
    forecast_one_step = make_har_rv_forecaster(prepared$data, predictive_draws),
    update_state = update_har_state,
    refit_every = refit_every,
    model_name = "harRV",
    save_fits = save_fits,
    seed = seed
  )
}

if (isTRUE(getOption("benchmark.har_autorun", FALSE))) {
  bench <- load_benchmark_data()
  if (!exists("benchmark_results")) benchmark_results <- list()
  benchmark_results$harRV <- run_har_rv(bench = bench)
}
