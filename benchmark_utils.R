default_benchmark_target <- function() "y_IQQH_EUR"

default_benchmark_covariates <- function() {
  c(
    "GCPU_baseline",
    "l_t",
    "c_t",
    "itraxx",
    "Term_Spread",
    "Brent_return",
    "TTF_return",
    "VSTOXX",
    "COVID_dummy",
    "Energy_crisis_dummy"
  )
}

default_dummy_covariates <- function() c("COVID_dummy", "Energy_crisis_dummy")

qlike <- function(actual_variance, forecast_variance) {
  eps <- .Machine$double.eps
  actual_variance <- pmax(as.numeric(actual_variance), eps)
  forecast_variance <- pmax(as.numeric(forecast_variance), eps)
  mean(actual_variance / forecast_variance - log(actual_variance / forecast_variance) - 1,
       na.rm = TRUE)
}

read_benchmark_raw <- function(train_file = "train_dataset.csv",
                               valid_file = "valid_dataset.csv",
                               test_file = "test_dataset.csv") {
  train <- read.csv(train_file, stringsAsFactors = FALSE)
  valid <- read.csv(valid_file, stringsAsFactors = FALSE)
  test <- read.csv(test_file, stringsAsFactors = FALSE)

  train$Date <- as.Date(train$Date)
  valid$Date <- as.Date(valid$Date)
  test$Date <- as.Date(test$Date)
  train$.split <- "train"
  valid$.split <- "valid"
  test$.split <- "test"
  train$.split_row <- seq_len(nrow(train))
  valid$.split_row <- seq_len(nrow(valid))
  test$.split_row <- seq_len(nrow(test))

  rbind(train, valid, test)
}

standardize_covariates <- function(raw,
                                   covariates = default_benchmark_covariates(),
                                   dummy_covariates = default_dummy_covariates(),
                                   scaling_rows = NULL) {
  scale_cols <- setdiff(covariates, dummy_covariates)
  scaling <- data.frame(
    variable = scale_cols,
    center = NA_real_,
    scale = NA_real_,
    stringsAsFactors = FALSE
  )

  if (is.null(scaling_rows)) scaling_rows <- raw$.split == "train"
  if (length(scaling_rows) != nrow(raw) || any(is.na(scaling_rows))) {
    stop("scaling_rows must be a non-missing logical vector with one entry per row.")
  }
  for (i in seq_along(scale_cols)) {
    col <- scale_cols[i]
    center <- mean(raw[scaling_rows, col], na.rm = TRUE)
    spread <- stats::sd(raw[scaling_rows, col], na.rm = TRUE)
    if (!is.finite(spread) || spread == 0) spread <- 1

    raw[[col]] <- (raw[[col]] - center) / spread
    scaling$center[i] <- center
    scaling$scale[i] <- spread
  }

  list(data = raw, scaling = scaling)
}

load_benchmark_data <- function(train_file = "train_dataset.csv",
                                valid_file = "valid_dataset.csv",
                                test_file = "test_dataset.csv",
                                target_col = default_benchmark_target(),
                                covariates = default_benchmark_covariates(),
                                dummy_covariates = default_dummy_covariates()) {

  raw <- read_benchmark_raw(train_file, valid_file, test_file)

  keep_cols <- c(target_col, covariates)
  keep <- stats::complete.cases(raw[, keep_cols, drop = FALSE])
  scaling_rows <- raw$.split == "train" & keep

  scaled <- standardize_covariates(
    raw, covariates, dummy_covariates, scaling_rows
  )
  raw <- scaled$data
  data <- raw[keep, , drop = FALSE]

  X_all <- as.matrix(data[, covariates, drop = FALSE])
  colnames(X_all) <- covariates
  y_all <- as.numeric(data[[target_col]])

  split_index <- split(seq_len(nrow(data)), data$.split)
  split_rows <- lapply(c("train", "valid", "test"), function(x) split_index[[x]])
  names(split_rows) <- c("train", "valid", "test")

  list(
    raw = raw,
    all = data,
    target_col = target_col,
    covariates = covariates,
    covariates_are_lagged = TRUE,
    x_cols = covariates,
    scaling = scaled$scaling,
    split_rows = split_rows,
    train = data[split_rows$train, , drop = FALSE],
    valid = data[split_rows$valid, , drop = FALSE],
    test = data[split_rows$test, , drop = FALSE],
    X_all = X_all,
    y_all = y_all,
    X_train = X_all[split_rows$train, , drop = FALSE],
    y_train = y_all[split_rows$train],
    X_valid = X_all[split_rows$valid, , drop = FALSE],
    y_valid = y_all[split_rows$valid],
    X_test = X_all[split_rows$test, , drop = FALSE],
    y_test = y_all[split_rows$test]
  )
}

# -------------------------------------------------------------------------
# Expanding-window, one-step-ahead forecasting framework
# -------------------------------------------------------------------------
#
# Timing convention
# -----------------
# For a forecast recorded on row t, the model is fitted using rows 1:(t - 1)
# and produces p(y_t | I_{t-1}).  Once y_t is observed, update_state() is
# called before the next forecast.  Therefore forecasts inside a 13-week
# refit block are still genuine one-step-ahead forecasts; they are not a
# sequence of 2-, ..., 13-step recursive forecasts.
#
# Callback contract
# -----------------
# fit_model(train_indices, previous_model, refit_id, bench) returns a list
# containing at least `model` and `state`; `fit` is optional and is retained
# for diagnostics.  forecast_one_step(model, state, forecast_index, bench)
# returns `variance_draws`, `return_draws`, and `state_prior`.  It may also
# return `log_density_draws`, either a numeric vector of log p(y_t | draw)
# or a function that accepts the realised y_t and returns that vector.
# update_state(model, state, forecast, observed_y, observation_index, bench)
# returns the updated state without re-estimating model parameters.

log_mean_exp <- function(log_x) {
  log_x <- as.numeric(log_x)
  log_x <- log_x[is.finite(log_x)]
  if (!length(log_x)) return(NA_real_)
  anchor <- max(log_x)
  anchor + log(mean(exp(log_x - anchor)))
}

summarize_predictive_draws <- function(forecast, observed_y) {
  return_draws <- as.numeric(forecast$return_draws)
  variance_draws <- as.numeric(forecast$variance_draws)

  if (length(return_draws) == 0L || length(variance_draws) == 0L) {
    stop("forecast_one_step() must return non-empty return_draws and variance_draws.")
  }
  if (any(!is.finite(return_draws)) || any(!is.finite(variance_draws))) {
    stop("forecast_one_step() returned non-finite predictive draws.")
  }

  log_density_draws <- forecast$log_density_draws
  if (is.function(log_density_draws)) {
    log_density_draws <- log_density_draws(observed_y)
  }
  variance <- safe_forecast_variance(mean(variance_draws))
  q01 <- as.numeric(stats::quantile(return_draws, 0.01, names = FALSE))
  q05 <- as.numeric(stats::quantile(return_draws, 0.05, names = FALSE))

  list(
    variance = variance,
    volatility = sqrt(variance),
    VaR_01 = q01,
    ES_01 = mean(return_draws[return_draws <= q01]),
    VaR_05 = q05,
    ES_05 = mean(return_draws[return_draws <= q05]),
    log_score = if (is.null(log_density_draws)) NA_real_
                else log_mean_exp(log_density_draws)
  )
}

as_benchmark_forecasts <- function(bench, forecast_table) {
  split_forecasts <- function(split_name) {
    rows <- forecast_table[forecast_table$split == split_name, , drop = FALSE]
    rows <- rows[order(rows$row_index), , drop = FALSE]
    variance <- safe_forecast_variance(rows$variance)

    list(
      variance = variance,
      volatility = sqrt(variance),
      actual_variance = actual_return_variance(rows$actual_return),
      actual_return = rows$actual_return,
      VaR_01 = rows$VaR_01,
      ES_01 = rows$ES_01,
      VaR_05 = rows$VaR_05,
      ES_05 = rows$ES_05,
      log_score = rows$log_score,
      dates = rows$forecast_date,
      qlike = qlike(actual_return_variance(rows$actual_return), variance)
    )
  }

  valid <- split_forecasts("valid")
  test <- split_forecasts("test")

  list(
    forecast_valid = valid$variance,
    forecast_test = test$variance,
    var_valid = valid$variance,
    var_test = test$variance,
    vol_valid = valid$volatility,
    vol_test = test$volatility,
    VaR_01_valid = valid$VaR_01,
    ES_01_valid = valid$ES_01,
    VaR_05_valid = valid$VaR_05,
    ES_05_valid = valid$ES_05,
    VaR_01_test = test$VaR_01,
    ES_01_test = test$ES_01,
    VaR_05_test = test$VaR_05,
    ES_05_test = test$ES_05,
    log_score_valid = valid$log_score,
    log_score_test = test$log_score,
    qlike_valid = valid$qlike,
    qlike_test = test$qlike,
    dates_valid = valid$dates,
    dates_test = test$dates
  )
}

make_rolling_schedule <- function(bench,
                                  forecast_indices,
                                  refit_every = 13L) {
  forecast_indices <- sort(unique(as.integer(forecast_indices)))
  if (length(forecast_indices) == 0L) {
    stop("forecast_indices must contain at least one row.")
  }
  if (any(forecast_indices <= 1L)) {
    stop("Each forecast needs at least one preceding training observation.")
  }
  if (length(forecast_indices) > 1L && any(diff(forecast_indices) != 1L)) {
    stop("forecast_indices must be consecutive rows so each realised return can update the state.")
  }
  if (length(refit_every) != 1L || !is.finite(refit_every) || refit_every < 1L) {
    stop("refit_every must be a positive integer.")
  }

  data.frame(
    position = seq_along(forecast_indices),
    row_index = forecast_indices,
    refit_id = ((seq_along(forecast_indices) - 1L) %/% as.integer(refit_every)) + 1L,
    refit_now = ((seq_along(forecast_indices) - 1L) %% as.integer(refit_every)) == 0L,
    stringsAsFactors = FALSE
  )
}

rolling_forecast <- function(bench,
                             fit_model,
                             forecast_one_step,
                             update_state,
                             forecast_indices = sort(c(
                               bench$split_rows$valid,
                               bench$split_rows$test
                             )),
                             refit_every = 13L,
                             model_name = "model",
                             save_fits = FALSE,
                             seed = NULL) {
  if (!is.function(fit_model) || !is.function(forecast_one_step) ||
      !is.function(update_state)) {
    stop("fit_model, forecast_one_step, and update_state must all be functions.")
  }
  if (!is.null(seed)) set.seed(seed)

  schedule <- make_rolling_schedule(bench, forecast_indices, refit_every)
  forecast_rows <- vector("list", nrow(schedule))
  refit_diagnostics <- vector("list", max(schedule$refit_id))
  retained_fits <- if (isTRUE(save_fits)) vector("list", max(schedule$refit_id)) else NULL

  model <- NULL
  state <- NULL
  last_fit <- NULL
  stopped_early <- FALSE
  stop_refit_id <- NA_integer_

  for (i in seq_len(nrow(schedule))) {
    row <- schedule[i, , drop = FALSE]
    forecast_index <- row$row_index

    if (isTRUE(row$refit_now)) {
      train_indices <- seq_len(forecast_index - 1L)
      fitted <- fit_model(
        train_indices = train_indices,
        previous_model = model,
        refit_id = row$refit_id,
        bench = bench
      )
      if (!is.list(fitted) || is.null(fitted$model) || is.null(fitted$state)) {
        stop("fit_model() must return a list with model and state components.")
      }

      refit_diagnostics[[row$refit_id]] <- list(
        refit_id = row$refit_id,
        train_start = bench$all$Date[1L],
        train_end = bench$all$Date[forecast_index - 1L],
        train_T = length(train_indices),
        forecast_start = bench$all$Date[forecast_index],
        diagnostics = fitted$diagnostics,
        usable = fitted$usable
      )

      # A refit that fails the model's own convergence gate (fitted$usable)
      # stops the rolling forecast here rather than being silently accepted
      # or papered over -- no forecast is produced from an unconverged fit --
      # UNLESS benchmark.stop_on_non_convergence is turned off, in which case
      # it's used anyway (with usable = FALSE staying on the record in
      # refit_diagnostics above) so a bad refit degrades the run instead of
      # halting it. Note this is independent of benchmark.check_convergence:
      # that one controls whether diagnostics are computed at all (off means
      # fitted$usable is always TRUE, so this branch is simply never
      # reached); this one controls what happens once a real failure is
      # found. Forecasts from earlier, converged refits are always returned
      # when stopping (the trailing unfilled forecast_rows slots are simply
      # dropped by rbind() below); a failure on the very first refit with
      # stopping enabled means zero rows, which check_result() in
      # run_models.R correctly treats as "no usable result" for this model.
      # fit_model() implementations that don't set $usable (e.g. GARCH/HAR)
      # leave it NULL, which never stops or warns -- this is opt-in and
      # backward compatible.
      if (identical(fitted$usable, FALSE)) {
        stop_on_failure <- isTRUE(getOption("benchmark.stop_on_non_convergence", TRUE))
        warning(sprintf(
          "%s: refit %d (training through %s) failed its convergence check.%s",
          model_name, row$refit_id, as.character(bench$all$Date[forecast_index - 1L]),
          if (stop_on_failure) {
            sprintf(" Stopping the rolling forecast there; %d earlier forecast(s) are still returned.", i - 1L)
          } else {
            " Continuing anyway (benchmark.stop_on_non_convergence is off); its forecasts should be treated as unreliable -- check refit_diagnostics$usable."
          }
        ), call. = FALSE)
        if (stop_on_failure) {
          stopped_early <- TRUE
          stop_refit_id <- row$refit_id
          break
        }
      }

      model <- fitted$model
      state <- fitted$state
      last_fit <- fitted$fit
      if (isTRUE(save_fits)) retained_fits[[row$refit_id]] <- last_fit
    }

    realised_y <- bench$y_all[forecast_index]
    forecast <- forecast_one_step(
      model = model,
      state = state,
      forecast_index = forecast_index,
      bench = bench
    )
    summary <- summarize_predictive_draws(forecast, realised_y)

    forecast_rows[[i]] <- data.frame(
      model = model_name,
      row_index = forecast_index,
      forecast_date = bench$all$Date[forecast_index],
      split = bench$all$.split[forecast_index],
      refit_id = row$refit_id,
      refit_now = row$refit_now,
      estimation_end = bench$all$Date[forecast_index - 1L],
      actual_return = realised_y,
      actual_variance = realised_y^2,
      variance = summary$variance,
      volatility = summary$volatility,
      VaR_01 = summary$VaR_01,
      ES_01 = summary$ES_01,
      VaR_05 = summary$VaR_05,
      ES_05 = summary$ES_05,
      log_score = summary$log_score,
      stringsAsFactors = FALSE
    )

    # The realised y_t is available immediately after making the one-step
    # forecast.  No parameter estimation occurs here; only the model state
    # is filtered/updated for the following origin.
    state <- update_state(
      model = model,
      state = state,
      forecast = forecast,
      observed_y = realised_y,
      observation_index = forecast_index,
      bench = bench
    )
  }

  # A convergence failure on the very first refit leaves every forecast_rows
  # slot NULL, so do.call(rbind, ...) returns NULL rather than a 0-row
  # data.frame; as_benchmark_forecasts() isn't written to handle that, so
  # skip it here and let check_result() in run_models.R reject the
  # NULL/empty forecast_table with its own clear message instead of an
  # obscure subsetting error.
  forecast_table <- do.call(rbind, forecast_rows)
  has_forecasts <- is.data.frame(forecast_table) && nrow(forecast_table) > 0L
  list(
    model_name = model_name,
    fit = last_fit,
    model = model,
    rolling_fits = retained_fits,
    forecast_table = forecast_table,
    forecasts = if (has_forecasts) as_benchmark_forecasts(bench, forecast_table) else NULL,
    diagnostics = list(refits = refit_diagnostics),
    refit_every = as.integer(refit_every),
    stopped_early = stopped_early,
    stop_refit_id = stop_refit_id
  )
}

combine_rolling_forecasts <- function(bench,
                                      validation_result,
                                      test_result,
                                      model_name = validation_result$model_name) {
  forecast_table <- rbind(
    validation_result$forecast_table,
    test_result$forecast_table
  )
  forecast_table <- forecast_table[order(forecast_table$row_index), , drop = FALSE]

  list(
    model_name = model_name,
    fit = test_result$fit,
    model = test_result$model,
    rolling_fits = list(
      validation = validation_result$rolling_fits,
      test = test_result$rolling_fits
    ),
    forecast_table = forecast_table,
    forecasts = as_benchmark_forecasts(bench, forecast_table),
    diagnostics = list(
      validation_refits = validation_result$diagnostics$refits,
      test_refits = test_result$diagnostics$refits
    ),
    refit_every = test_result$refit_every
  )
}

select_rolling_architecture <- function(bench,
                                        candidates,
                                        run_candidate,
                                        metric = c("qlike", "negative_log_score"),
                                        refit_every = 13L) {
  metric <- match.arg(metric)
  if (!is.list(candidates) || length(candidates) == 0L || !is.function(run_candidate)) {
    stop("candidates must be a non-empty list and run_candidate must be a function.")
  }

  validation_indices <- bench$split_rows$valid
  # 2026-08-26: one fit per candidate over the whole validation window, not
  # the refit_every-week rolling refit -- picking an architecture doesn't
  # need the quarterly re-estimation cadence, and with 4 candidates that
  # cadence would mean ~4-5 Stan refits per candidate just to rank them.
  # refit_every itself is untouched below, so the final test_run on the
  # selected candidate still re-estimates on the real schedule.
  validation_refit_every <- max(length(validation_indices), 1L)
  validation_runs <- lapply(seq_along(candidates), function(i) {
    run_candidate(candidates[[i]], validation_indices, validation_refit_every)
  })
  scores <- vapply(validation_runs, function(x) {
    if (metric == "qlike") {
      x$forecasts$qlike_valid
    } else {
      -mean(x$forecasts$log_score_valid, na.rm = TRUE)
    }
  }, numeric(1))
  if (all(!is.finite(scores))) {
    stop("No candidate produced a finite validation score.")
  }

  # Divergent transitions from the validation-window fit(s)
  # (refit_diagnostics[[j]]$diagnostics$divergence_sum -- see
  # stan_fit_diagnostics()), summed per candidate. QLIKE alone can't tell a
  # real architecture win from a candidate that just happened to land in a
  # good spot on a struggling sampler, so a candidate only competes on score
  # if its validation fit was divergence-free.
  divergences <- vapply(validation_runs, function(x) {
    refits <- x$diagnostics$refits
    if (is.null(refits) || length(refits) == 0L) return(NA_real_)
    per_refit <- vapply(refits, function(r) {
      d <- r$diagnostics$divergence_sum
      if (is.null(d) || !is.finite(d)) NA_real_ else d
    }, numeric(1))
    if (all(is.na(per_refit))) NA_real_ else sum(per_refit, na.rm = TRUE)
  }, numeric(1))
  clean <- is.finite(divergences) & divergences == 0

  # Lexicographic selection: divergence count first, QLIKE second. Restrict
  # to whichever candidates share the *lowest* divergence count -- when at
  # least one is clean that's just "the clean ones" (min = 0), same as
  # before; when every candidate diverged somewhere, it's now the
  # least-divergent tier instead of throwing the divergence count away and
  # ranking everyone by score. Only candidates with no divergence diagnostics
  # at all (finite_divergence all FALSE -- diagnostics missing entirely, not
  # just nonzero) fall back to considering everyone, so this never picks
  # nothing.
  finite_divergence <- is.finite(divergences)
  if (any(finite_divergence)) {
    min_divergence <- min(divergences[finite_divergence])
    eligible <- finite_divergence & divergences == min_divergence
    if (min_divergence > 0) {
      labels <- vapply(candidates, function(x) x$label %||% paste0("K", x$K), character(1))
      warning("select_rolling_architecture: no candidate was divergence-free in the ",
              "validation window (", paste(labels, divergences, sep = "=", collapse = ", "),
              "); restricting to the ", sum(eligible),
              " candidate(s) tied at the lowest divergence count (", min_divergence,
              ") and ranking those by score.", call. = FALSE)
    }
  } else {
    labels <- vapply(candidates, function(x) x$label %||% paste0("K", x$K), character(1))
    warning("select_rolling_architecture: no candidate reported validation-window ",
            "divergence diagnostics (", paste(labels, collapse = ", "),
            "); selecting by score alone.", call. = FALSE)
    eligible <- rep(TRUE, length(candidates))
  }
  selected_index <- which.min(replace(scores, !is.finite(scores) | !eligible, Inf))

  test_run <- run_candidate(
    candidates[[selected_index]],
    bench$split_rows$test,
    refit_every
  )
  final <- combine_rolling_forecasts(
    bench,
    validation_runs[[selected_index]],
    test_run,
    model_name = test_run$model_name
  )
  final$architecture_selection <- list(
    metric = metric,
    candidates = candidates,
    validation_scores = scores,
    validation_divergences = divergences,
    validation_clean = clean,
    selected_index = selected_index,
    selected = candidates[[selected_index]]
  )
  final
}

actual_return_variance <- function(y) {
  as.numeric(y)^2
}

safe_forecast_variance <- function(x) {
  pmax(as.numeric(x), .Machine$double.eps)
}

finite_reduce <- function(x, fn) {
  x <- as.numeric(x[is.finite(x)])
  if (length(x)) fn(x) else NA_real_
}

stan_fit_diagnostics <- function(fit, pars = NULL, rhat_limit = 1.01,
                                 ess_limit = 400, ebfmi_limit = 0.30) {
  if (!isTRUE(getOption("benchmark.check_convergence", TRUE))) {
    checks <- list(
      rhat_ok = TRUE,
      bulk_ess_ok = TRUE,
      tail_ess_ok = TRUE,
      divergences_ok = TRUE,
      ebfmi_ok = TRUE
    )
    return(list(
      skipped = TRUE,
      max_rhat = NA_real_,
      min_bulk_ess = NA_real_,
      min_tail_ess = NA_real_,
      divergence_sum = NA_integer_,
      min_ebfmi = NA_real_,
      rhat = numeric(),
      bulk_ess = numeric(),
      tail_ess = numeric(),
      parameter_table = data.frame(),
      divergence_by_chain = numeric(),
      ebfmi_by_chain = numeric(),
      checks = checks,
      rhat_pass = TRUE,
      bulk_ess_pass = TRUE,
      tail_ess_pass = TRUE,
      divergence_pass = TRUE,
      ebfmi_pass = TRUE,
      all_checks_passed = TRUE,
      thresholds = list(
        rhat = rhat_limit,
        bulk_ess = ess_limit,
        tail_ess = ess_limit,
        ebfmi = ebfmi_limit
      )
    ))
  }

  # lp__ is a sampler quantity, not a declared model parameter -- it is never
  # in fit@model_pars, so a plain intersect(pars, fit@model_pars) silently
  # drops it even when the caller explicitly asked for it. Special-case it
  # back in; rstan::extract() accepts "lp__" directly regardless of
  # fit@model_pars, and everything below treats it like any other variable.
  available_pars <- if (is.null(pars)) {
    fit@model_pars
  } else {
    c(intersect(pars, fit@model_pars), intersect(pars, "lp__"))
  }
  raw_draws <- rstan::extract(fit, pars = available_pars, permuted = FALSE,
                              inc_warmup = FALSE)
  parameter_summary <- posterior::summarise_draws(
    posterior::as_draws_array(raw_draws),
    rhat = posterior::rhat,
    ess_bulk = posterior::ess_bulk,
    ess_tail = posterior::ess_tail
  )
  parameter_summary <- as.data.frame(parameter_summary)
  rhat <- stats::setNames(parameter_summary$rhat, parameter_summary$variable)
  bulk_ess <- stats::setNames(parameter_summary$ess_bulk,
                              parameter_summary$variable)
  tail_ess <- stats::setNames(parameter_summary$ess_tail,
                              parameter_summary$variable)

  sampler_params <- rstan::get_sampler_params(fit, inc_warmup = FALSE)

  divergence_by_chain <- vapply(
    sampler_params,
    function(chain) sum(chain[, "divergent__"] > 0, na.rm = TRUE),
    numeric(1)
  )
  names(divergence_by_chain) <- paste0("chain_", seq_along(sampler_params))
  divergence_sum <- sum(divergence_by_chain)
  ebfmi_by_chain <- tryCatch(
    as.numeric(rstan::get_bfmi(fit)),
    error = function(e) {
      warning("E-BFMI could not be calculated: ", conditionMessage(e))
      rep(NA_real_, length(sampler_params))
    }
  )
  names(ebfmi_by_chain) <- paste0("chain_", seq_along(sampler_params))

  max_rhat <- finite_reduce(rhat, max)
  min_bulk_ess <- finite_reduce(bulk_ess, min)
  min_tail_ess <- finite_reduce(tail_ess, min)
  min_ebfmi <- finite_reduce(ebfmi_by_chain, min)

  checks <- list(
    rhat_ok = is.finite(max_rhat) && max_rhat < rhat_limit,
    bulk_ess_ok = is.finite(min_bulk_ess) && min_bulk_ess > ess_limit,
    tail_ess_ok = is.finite(min_tail_ess) && min_tail_ess > ess_limit,
    divergences_ok = divergence_sum == 0L,
    ebfmi_ok = is.finite(min_ebfmi) && min_ebfmi > ebfmi_limit
  )

  list(
    max_rhat = max_rhat,
    min_bulk_ess = min_bulk_ess,
    min_tail_ess = min_tail_ess,
    divergence_sum = divergence_sum,
    min_ebfmi = min_ebfmi,

    rhat = rhat,
    bulk_ess = bulk_ess,
    tail_ess = tail_ess,
    parameter_table = parameter_summary,

    divergence_by_chain = divergence_by_chain,
    ebfmi_by_chain = ebfmi_by_chain,

    checks = checks,

    rhat_pass = checks$rhat_ok,
    bulk_ess_pass = checks$bulk_ess_ok,
    tail_ess_pass = checks$tail_ess_ok,
    ebfmi_pass = checks$ebfmi_ok,

    all_checks_passed = all(unlist(checks)),

    thresholds = list(
      rhat = rhat_limit,
      bulk_ess = ess_limit,
      tail_ess = ess_limit,
      ebfmi = ebfmi_limit
    )
  )
}


assert_stan_diagnostics <- function(diagnostics, model_name = "Stan model",
                                    strict = FALSE) {
  problems <- character()

  if (!isTRUE(diagnostics$checks$rhat_ok)) {
    problems <- c(problems, sprintf("max R-hat = %.4f, required < 1.01",
                                    diagnostics$max_rhat))
  }
  if (!isTRUE(diagnostics$checks$bulk_ess_ok)) {
    problems <- c(problems, sprintf("min bulk-ESS = %.1f, required > 400",
                                    diagnostics$min_bulk_ess))
  }
  if (!isTRUE(diagnostics$checks$tail_ess_ok)) {
    problems <- c(problems, sprintf("min tail-ESS = %.1f, required > 400",
                                    diagnostics$min_tail_ess))
  }
  if (!isTRUE(diagnostics$checks$divergences_ok)) {
    problems <- c(problems, sprintf("%d divergent transitions",
                                    diagnostics$divergence_sum))
  }
  if (!isTRUE(diagnostics$checks$ebfmi_ok)) {
    problems <- c(problems, sprintf("min E-BFMI = %.3f, required > 0.30",
                                    diagnostics$min_ebfmi))
  }

  if (length(problems)) {
    diagnostic_message <- paste0(model_name, " diagnostics failed: ",
                                 paste(problems, collapse = "; "), ".")
    if (isTRUE(strict)) stop(diagnostic_message, call. = FALSE)
    warning(diagnostic_message, call. = FALSE)
  }

  invisible(diagnostics)
}



resolve_stan_plot_variables <- function(requested, draw_variables) {
  requested <- unique(as.character(requested))
  draw_variables <- as.character(draw_variables)
  selected <- character()

  for (parameter in requested) {
    exact <- draw_variables[draw_variables == parameter]
    indexed <- draw_variables[startsWith(draw_variables, paste0(parameter, "["))]
    selected <- c(selected, exact, indexed)
  }

  unique(selected)
}

save_stan_diagnostic_plots <- function(fit, file, pars, pairs_pars = NULL) {
  if (!inherits(fit, "stanfit")) {
    stop("fit must be an rstan stanfit object.", call. = FALSE)
  }

  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    warning("Package 'bayesplot' is not installed; plots were skipped.",
            call. = FALSE)
    return(invisible(FALSE))
  }

  requested_base_pars <- sub("\\[.*$", "", pars)
  extract_pars <- intersect(unique(requested_base_pars), fit@model_pars)

  if (length(extract_pars) == 0L) {
    warning("None of the requested plotting parameters exists in fit.",
            call. = FALSE)
    return(invisible(FALSE))
  }

  draws_array <- rstan::extract(fit, pars = extract_pars, permuted = FALSE,
                                inc_warmup = FALSE)
  available_pars <- resolve_stan_plot_variables(
    requested = pars,
    draw_variables = dimnames(draws_array)[[3L]]
  )

  if (length(available_pars) == 0L) {
    warning("None of the requested Stan diagnostic variables exists in fit.",
            call. = FALSE)
    return(invisible(FALSE))
  }

  if (is.null(pairs_pars)) {
    pairs_pars <- head(available_pars, 4L)
  } else {
    pairs_pars <- resolve_stan_plot_variables(
      requested = pairs_pars,
      draw_variables = available_pars
    )
  }

  grDevices::pdf(file, width = 11, height = 8)
  on.exit(grDevices::dev.off(), add = TRUE)

  print(bayesplot::mcmc_trace(draws_array, pars = available_pars))
  print(bayesplot::mcmc_rank_hist(draws_array, pars = available_pars))

  if (length(pairs_pars) >= 2L) {
    print(bayesplot::mcmc_pairs(draws_array, pars = pairs_pars))
  }

  invisible(TRUE)
}
