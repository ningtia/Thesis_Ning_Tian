# Model comparison based on the dated forecast_table returned by each runner.

source("benchmark_utils.R")

`%||%` <- function(x, y) if (is.null(x)) y else x

model_test_forecasts <- function(result, model_name) {
  forecast_table <- result$forecast_table
  if (is.null(forecast_table)) {
    stop(sprintf("%s has no forecast_table; rerun it with rolling_forecast().", model_name))
  }
  rows <- forecast_table[forecast_table$split == "test", , drop = FALSE]
  if (!nrow(rows)) stop(sprintf("%s has no test-period forecasts.", model_name))
  rows[order(rows$forecast_date), , drop = FALSE]
}

available_forecast_models <- function(benchmark_results) {
  names(Filter(function(x) !is.null(x$forecast_table), benchmark_results))
}

fz0_score <- function(actual_return, VaR, ES, alpha) {
  valid <- is.finite(actual_return) & is.finite(VaR) & is.finite(ES) &
    ES < 0 & ES <= VaR
  score <- rep(NA_real_, length(actual_return))
  hit <- as.numeric(actual_return <= VaR)
  score[valid] <- -hit[valid] * (VaR[valid] - actual_return[valid]) /
    (alpha * ES[valid]) + VaR[valid] / ES[valid] + log(-ES[valid]) - 1
  score
}

qlike_loss <- function(actual_return, forecast_variance) {
  eps <- .Machine$double.eps
  actual_variance <- pmax(as.numeric(actual_return)^2, eps)
  forecast_variance <- pmax(as.numeric(forecast_variance), eps)
  actual_variance / forecast_variance - log(actual_variance / forecast_variance) - 1
}

cpu_regime <- function(bench, cpu_variable = "GCPU_baseline") {
  if (!cpu_variable %in% names(bench$all)) {
    stop(sprintf("CPU variable '%s' is absent from bench$all.", cpu_variable))
  }
  threshold <- stats::median(bench$all[[cpu_variable]][bench$split_rows$train],
                             na.rm = TRUE)
  if (!is.finite(threshold)) stop("Unable to calculate the training-sample CPU threshold.")

  data.frame(
    forecast_date = bench$all$Date,
    CPU = bench$all[[cpu_variable]],
    CPU_regime = ifelse(bench$all[[cpu_variable]] >= threshold, "high", "low"),
    stringsAsFactors = FALSE
  )
}

density_score_table <- function(bench, benchmark_results,
                                cpu_variable = "GCPU_baseline") {
  regimes <- cpu_regime(bench, cpu_variable)
  rows <- lapply(available_forecast_models(benchmark_results), function(model_name) {
    forecast <- model_test_forecasts(benchmark_results[[model_name]], model_name)
    forecast <- merge(forecast, regimes[, c("forecast_date", "CPU_regime")],
                      by = "forecast_date", all.x = TRUE, sort = FALSE)
    log_score <- forecast$log_score
    data.frame(
      Model = model_name,
      N = sum(is.finite(log_score)),
      LPS_overall = mean(log_score, na.rm = TRUE),
      LPS_CPU_low = mean(log_score[forecast$CPU_regime == "low"], na.rm = TRUE),
      LPS_CPU_high = mean(log_score[forecast$CPU_regime == "high"], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

xlogy <- function(x, y) {
  out <- numeric(length(x))
  positive <- x > 0
  out[positive] <- x[positive] * log(y[positive])
  out
}

kupiec_uc_test <- function(hits, alpha) {
  hits <- as.integer(hits)
  n_obs <- length(hits)
  n_hit <- sum(hits)
  if (n_obs == 0L) return(list(statistic = NA_real_, p_value = NA_real_))
  p_hat <- n_hit / n_obs
  log_l0 <- xlogy(n_hit, alpha) + xlogy(n_obs - n_hit, 1 - alpha)
  log_l1 <- xlogy(n_hit, p_hat) + xlogy(n_obs - n_hit, 1 - p_hat)
  statistic <- max(0, -2 * (log_l0 - log_l1))
  list(statistic = statistic, p_value = stats::pchisq(statistic, df = 1, lower.tail = FALSE))
}

christoffersen_independence_test <- function(hits) {
  hits <- as.integer(hits)
  if (length(hits) < 3L) return(list(statistic = NA_real_, p_value = NA_real_))
  previous <- hits[-length(hits)]
  current <- hits[-1L]
  n00 <- sum(previous == 0L & current == 0L)
  n01 <- sum(previous == 0L & current == 1L)
  n10 <- sum(previous == 1L & current == 0L)
  n11 <- sum(previous == 1L & current == 1L)

  p01 <- if ((n00 + n01) == 0L) 0 else n01 / (n00 + n01)
  p11 <- if ((n10 + n11) == 0L) 0 else n11 / (n10 + n11)
  p <- (n01 + n11) / (n00 + n01 + n10 + n11)
  log_l_independent <- xlogy(n01 + n11, p) + xlogy(n00 + n10, 1 - p)
  log_l_markov <- xlogy(n01, p01) + xlogy(n00, 1 - p01) +
    xlogy(n11, p11) + xlogy(n10, 1 - p11)
  statistic <- max(0, -2 * (log_l_independent - log_l_markov))
  list(statistic = statistic, p_value = stats::pchisq(statistic, df = 1, lower.tail = FALSE))
}

dynamic_quantile_test <- function(hits, VaR, alpha, lags = 4L) {
  hits <- as.numeric(hits)
  VaR <- as.numeric(VaR)
  n_obs <- length(hits)
  if (n_obs <= lags + 5L || all(hits == 0) || all(hits == 1)) {
    return(list(statistic = NA_real_, p_value = NA_real_, df = NA_integer_))
  }

  dependent <- hits[(lags + 1L):n_obs] - alpha
  lag_matrix <- sapply(seq_len(lags), function(j) {
    hits[(lags + 1L - j):(n_obs - j)] - alpha
  })
  if (is.null(dim(lag_matrix))) lag_matrix <- matrix(lag_matrix, ncol = 1L)
  scaled_var <- as.numeric(scale(VaR[(lags + 1L):n_obs]))
  design <- cbind(1, lag_matrix, scaled_var)
  fit <- stats::lm(dependent ~ design - 1)
  statistic <- length(dependent) * summary(fit)$r.squared
  degrees_freedom <- ncol(design)
  list(
    statistic = statistic,
    p_value = stats::pchisq(statistic, df = degrees_freedom, lower.tail = FALSE),
    df = degrees_freedom
  )
}

tail_backtest_table <- function(benchmark_results, alphas = c(0.01, 0.05)) {
  rows <- list()
  counter <- 1L
  for (model_name in available_forecast_models(benchmark_results)) {
    forecast <- model_test_forecasts(benchmark_results[[model_name]], model_name)
    for (alpha in alphas) {
      suffix <- sprintf("%02d", as.integer(100 * alpha))
      VaR <- forecast[[paste0("VaR_", suffix)]]
      ES <- forecast[[paste0("ES_", suffix)]]
      if (is.null(VaR) || is.null(ES)) next
      valid <- complete.cases(forecast$actual_return, VaR, ES)
      actual <- forecast$actual_return[valid]
      VaR <- VaR[valid]
      ES <- ES[valid]
      hits <- as.integer(actual < VaR)
      uc <- kupiec_uc_test(hits, alpha)
      independence <- christoffersen_independence_test(hits)
      cc_statistic <- uc$statistic + independence$statistic
      dq <- dynamic_quantile_test(hits, VaR, alpha)
      fz <- fz0_score(actual, VaR, ES, alpha)

      rows[[counter]] <- data.frame(
        Model = model_name,
        alpha = alpha,
        N = length(actual),
        expected_exceedances = length(actual) * alpha,
        actual_exceedances = sum(hits),
        exceedance_rate = mean(hits),
        LR_UC = uc$statistic,
        p_UC = uc$p_value,
        LR_IND = independence$statistic,
        p_IND = independence$p_value,
        LR_CC = cc_statistic,
        p_CC = stats::pchisq(cc_statistic, df = 2, lower.tail = FALSE),
        DQ = dq$statistic,
        p_DQ = dq$p_value,
        FZ0_mean = mean(fz, na.rm = TRUE),
        mean_forecast_ES = mean(ES, na.rm = TRUE),
        realised_ES = if (any(hits == 1L)) mean(actual[hits == 1L]) else NA_real_,
        stringsAsFactors = FALSE
      )
      counter <- counter + 1L
    }
  }
  do.call(rbind, rows)
}

loss_matrix <- function(benchmark_results,
                        loss = c("qlike", "negative_lps", "fz05")) {
  loss <- match.arg(loss)
  model_names <- available_forecast_models(benchmark_results)
  series <- lapply(model_names, function(model_name) {
    forecast <- model_test_forecasts(benchmark_results[[model_name]], model_name)
    value <- switch(
      loss,
      qlike = qlike_loss(forecast$actual_return, forecast$variance),
      negative_lps = -forecast$log_score,
      fz05 = fz0_score(forecast$actual_return, forecast$VaR_05, forecast$ES_05, 0.05)
    )
    stats::setNames(value, as.character(forecast$forecast_date))
  })
  names(series) <- model_names
  common_dates <- Reduce(intersect, lapply(series, names))
  if (length(common_dates) == 0L) stop("No common out-of-sample dates across models.")
  loss_values <- sapply(series, function(x) x[common_dates])
  if (is.null(dim(loss_values))) loss_values <- matrix(loss_values, ncol = 1L)
  colnames(loss_values) <- model_names
  complete <- stats::complete.cases(loss_values)
  list(
    dates = as.Date(common_dates[complete]),
    loss = loss_values[complete, , drop = FALSE]
  )
}

newey_west_lrv <- function(x, max_lag = NULL) {
  x <- as.numeric(x)
  n_obs <- length(x)
  if (is.null(max_lag)) max_lag <- floor(4 * (n_obs / 100)^(2 / 9))
  x <- x - mean(x)
  gamma0 <- sum(x^2) / n_obs
  if (max_lag == 0L) return(gamma0)
  lags <- seq_len(min(max_lag, n_obs - 1L))
  gamma <- vapply(lags, function(j) sum(x[(j + 1L):n_obs] * x[1:(n_obs - j)]) / n_obs,
                  numeric(1))
  gamma0 + 2 * sum((1 - lags / (max_lag + 1)) * gamma)
}

dm_test_pair <- function(loss_a, loss_b, name_a, name_b, max_lag = NULL) {
  valid <- is.finite(loss_a) & is.finite(loss_b)
  difference <- loss_a[valid] - loss_b[valid]
  n_obs <- length(difference)
  if (n_obs < 10L) {
    return(data.frame(Model_A = name_a, Model_B = name_b, N = n_obs,
                      mean_loss_A_minus_B = NA_real_, statistic = NA_real_,
                      p_value = NA_real_, better_model = NA_character_))
  }
  lrv <- newey_west_lrv(difference, max_lag)
  statistic <- if (is.finite(lrv) && lrv > 0) mean(difference) / sqrt(lrv / n_obs) else NA_real_
  data.frame(
    Model_A = name_a,
    Model_B = name_b,
    N = n_obs,
    mean_loss_A_minus_B = mean(difference),
    statistic = statistic,
    p_value = 2 * stats::pnorm(-abs(statistic)),
    better_model = ifelse(mean(difference) < 0, name_a, name_b),
    stringsAsFactors = FALSE
  )
}

pairwise_dm_tests <- function(benchmark_results, loss = "qlike", max_lag = NULL) {
  losses <- loss_matrix(benchmark_results, loss)
  model_names <- colnames(losses$loss)
  if (length(model_names) < 2L) return(data.frame())
  pairs <- utils::combn(model_names, 2L, simplify = FALSE)
  result <- do.call(rbind, lapply(pairs, function(pair) {
    dm_test_pair(
      losses$loss[, pair[1L]], losses$loss[, pair[2L]],
      pair[1L], pair[2L], max_lag
    )
  }))
  result$loss <- loss
  result
}

model_confidence_set <- function(benchmark_results,
                                 loss = "qlike",
                                 alpha = 0.05,
                                 B = 5000L,
                                 statistic = "Tmax") {
  if (!requireNamespace("MCS", quietly = TRUE)) {
    return(list(available = FALSE, object = NULL,
                message = "Package 'MCS' is not installed; MCS was skipped."))
  }
  losses <- loss_matrix(benchmark_results, loss)$loss
  if (ncol(losses) < 2L) {
    return(list(available = FALSE, object = NULL, message = "At least two models are required."))
  }
  result <- tryCatch(
    MCS::MCSprocedure(
      Loss = losses, alpha = alpha, B = as.integer(B),
      statistic = statistic, verbose = FALSE
    ),
    error = function(e) e
  )
  if (inherits(result, "error")) {
    return(list(available = FALSE, object = NULL, message = conditionMessage(result)))
  }
  list(available = TRUE, object = result, message = NULL)
}

extract_pareto_k <- function(loo_object) {
  k <- loo_object$diagnostics$pareto_k %||% NULL
  if (is.null(k) && requireNamespace("loo", quietly = TRUE)) {
    k <- tryCatch(loo::pareto_k_values(loo_object), error = function(e) NULL)
  }
  as.numeric(k)
}

stanfit_psis_loo <- function(fit, k_threshold = 0.7) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    return(list(available = FALSE, loo = NULL, message = "Package 'loo' is not installed."))
  }
  if (!inherits(fit, "stanfit")) {
    return(list(available = FALSE, loo = NULL, message = "The stored fit is not a stanfit object."))
  }

  # rstan's loo method automatically applies moment matching to Pareto-k
  # observations above k_threshold while the compiled model is in this session.
  loo_object <- tryCatch(
    loo::loo(
      fit, pars = "log_lik", moment_match = TRUE,
      k_threshold = k_threshold, r_eff = TRUE
    ),
    error = function(e) e
  )
  moment_matching_applied <- !inherits(loo_object, "error")

  # 2026-08-27: moment matching needs the COMPILED model, not just the draws --
  # it calls log_prob()/grad_log_prob() on the stanfit. After a
  # saveRDS()/readRDS() round trip the DSO pointer is stale and rstan aborts
  # with "the model object is not created or not valid", which is exactly what
  # happened when this ran over results/models/*.rds in a fresh session: both
  # SV models reported available = FALSE and the thesis lost its whole LOO
  # comparison. Plain PSIS-LOO needs only the log_lik draws, which survive
  # serialisation intact, so fall back to it rather than returning nothing.
  # The fallback is recorded (moment_matching_applied) instead of being
  # silently equivalent: without moment matching, high-k observations are
  # approximated less well and must be reported as such.
  fallback_reason <- NULL
  if (!moment_matching_applied) {
    fallback_reason <- conditionMessage(loo_object)
    loo_object <- tryCatch({
      log_lik <- loo::extract_log_lik(fit, "log_lik", merge_chains = FALSE)
      loo::loo(log_lik, r_eff = loo::relative_eff(exp(log_lik)))
    }, error = function(e) e)
  }

  if (inherits(loo_object, "error")) {
    return(list(
      available = FALSE, loo = NULL,
      message = paste0(
        "PSIS-LOO failed. Moment matching: ",
        if (is.null(fallback_reason)) "not attempted" else fallback_reason,
        ". Plain PSIS: ", conditionMessage(loo_object)
      )
    ))
  }

  pareto_k <- extract_pareto_k(loo_object)
  problematic <- which(is.finite(pareto_k) & pareto_k > k_threshold)
  list(
    available = TRUE,
    loo = loo_object,
    pareto_k = pareto_k,
    problematic = problematic,
    moment_matching_requested = TRUE,
    moment_matching_applied = moment_matching_applied,
    fallback_reason = fallback_reason,
    message = if (!moment_matching_applied && length(problematic)) {
      paste0("Plain PSIS-LOO (moment matching unavailable: ", fallback_reason,
             "); high Pareto-k observations remain and are reported, not corrected.")
    } else if (!moment_matching_applied) {
      paste0("Plain PSIS-LOO (moment matching unavailable: ", fallback_reason, ").")
    } else if (length(problematic)) {
      "High Pareto-k remains after moment matching; exact leave-one-out refits are required."
    } else {
      NULL
    }
  )
}

bayesian_loo_comparison <- function(benchmark_results, k_threshold = 0.7) {
  preferred <- c("nonlinearSV", "linearSV", "restrictedSV")
  present <- intersect(preferred, names(benchmark_results))
  # Current linearSV is the nested restriction of NN-SV with g_theta = 0.  Do
  # not duplicate it as a second identical restricted model unless a separately
  # estimated benchmark_results$restrictedSV is supplied.
  display_name <- c(
    nonlinearSV = "Nonlinear SV",
    linearSV = "Linear SV (g_theta = 0)",
    restrictedSV = "Restricted SV"
  )
  reports <- lapply(present, function(name) stanfit_psis_loo(
    benchmark_results[[name]]$fit, k_threshold
  ))
  names(reports) <- display_name[present]
  usable <- reports[vapply(reports, function(x) isTRUE(x$available), logical(1))]
  loo_list <- lapply(usable, `[[`, "loo")

  comparison <- if (length(loo_list) >= 2L) {
    tryCatch(as.data.frame(loo::loo_compare(loo_list)), error = function(e) e)
  } else {
    NULL
  }
  if (inherits(comparison, "error")) comparison <- data.frame(error = conditionMessage(comparison))
  stacking <- if (length(loo_list) >= 2L) {
    tryCatch(loo::loo_model_weights(loo_list, method = "stacking"), error = function(e) e)
  } else {
    NULL
  }
  if (inherits(stacking, "error")) stacking <- data.frame(error = conditionMessage(stacking))

  pareto_summary <- do.call(rbind, lapply(names(reports), function(name) {
    report <- reports[[name]]
    k <- report$pareto_k %||% numeric()
    data.frame(
      Model = name,
      available = isTRUE(report$available),
      n_high_k = sum(k > k_threshold, na.rm = TRUE),
      max_k = if (length(k)) max(k, na.rm = TRUE) else NA_real_,
      action = if (!isTRUE(report$available)) report$message else if (length(report$problematic)) {
        "Refit remaining high-k folds exactly"
      } else {
        "PSIS-LOO accepted (moment matching requested)"
      },
      stringsAsFactors = FALSE
    )
  }))

  # 2026-08-27: one tidy, writable row per model. Everything the thesis needs
  # to report a LOO comparison used to live only inside the S4/matrix objects
  # above, so nothing could be written to results/tables/ and the comparison
  # had to be recomputed by hand. elpd_diff/se_diff come straight from
  # loo_compare (best model first, its elpd_diff being 0 by construction).
  comparison_table <- NULL
  if (is.data.frame(comparison) && nrow(comparison) && "elpd_diff" %in% names(comparison)) {
    weights <- if (inherits(stacking, "numeric") || inherits(stacking, "stacking_weights")) {
      stats::setNames(as.numeric(stacking), names(usable))
    } else {
      stats::setNames(rep(NA_real_, length(usable)), names(usable))
    }
    # loo >= 2.6 returns loo_compare() with the model names in a `model`
    # COLUMN; older versions put them in rownames, and as.data.frame() on that
    # object then leaves rownames as "1", "2", ... Taking rownames
    # unconditionally silently produced a table whose Model column was row
    # numbers, which then failed every join below and NA'd out the stacking
    # weights and Pareto diagnostics.
    model_names <- if ("model" %in% names(comparison)) {
      as.character(comparison$model)
    } else {
      rownames(comparison)
    }
    comparison_table <- data.frame(
      Model = model_names,
      elpd_diff = comparison$elpd_diff,
      se_diff = comparison$se_diff,
      elpd_loo = if ("elpd_loo" %in% names(comparison)) comparison$elpd_loo else NA_real_,
      p_loo = if ("p_loo" %in% names(comparison)) comparison$p_loo else NA_real_,
      stringsAsFactors = FALSE
    )
    comparison_table$stacking_weight <- unname(weights[comparison_table$Model])
    matched <- match(comparison_table$Model, pareto_summary$Model)
    comparison_table$n_high_pareto_k <- pareto_summary$n_high_k[matched]
    comparison_table$max_pareto_k <- pareto_summary$max_k[matched]
    comparison_table$moment_matched <- vapply(
      comparison_table$Model,
      function(nm) isTRUE(reports[[nm]]$moment_matching_applied),
      logical(1)
    )
    row.names(comparison_table) <- NULL
  }

  list(
    reports = reports,
    loo_compare = comparison,
    stacking_weights = stacking,
    pareto_k = pareto_summary,
    comparison_table = comparison_table
  )
}

# A high-k exact-refit hook.  The model-specific callback must refit the model
# with the requested observation handled according to the research design and
# return its exact log predictive density.  This avoids silently treating a
# failed PSIS approximation as a valid LOO value.
refit_problematic_loo_folds <- function(loo_comparison, refit_fold) {
  if (!is.function(refit_fold)) stop("refit_fold must be a function.")
  rows <- list()
  counter <- 1L
  for (model_name in names(loo_comparison$reports)) {
    indices <- loo_comparison$reports[[model_name]]$problematic %||% integer()
    for (index in indices) {
      rows[[counter]] <- data.frame(
        Model = model_name,
        observation = index,
        exact_log_predictive_density = refit_fold(model_name, index),
        stringsAsFactors = FALSE
      )
      counter <- counter + 1L
    }
  }
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}

ppc_statistics <- function(y, acf_lags) {
  acf_values <- stats::acf(y^2, lag.max = max(acf_lags), plot = FALSE)$acf[-1L]
  c(
    mean_squared_return = mean(y^2),
    q95_squared_return = as.numeric(stats::quantile(y^2, 0.95)),
    mean_absolute_return = mean(abs(y)),
    q95_absolute_return = as.numeric(stats::quantile(abs(y), 0.95)),
    stats::setNames(acf_values[acf_lags], paste0("acf_squared_lag_", acf_lags))
  )
}

posterior_predictive_checks <- function(bench,
                                        benchmark_results,
                                        n_rep = 500L,
                                        acf_lags = 1:10,
                                        seed = 777L) {
  set.seed(seed)
  bayesian_models <- intersect(
    c("nonlinearSV", "linearSV", "regimeSwitchingSV"),
    names(benchmark_results)
  )
  all_rows <- list()
  counter <- 1L

  for (model_name in bayesian_models) {
    result <- benchmark_results[[model_name]]
    post <- result$model$posterior %||% result$posterior
    if (is.null(post) || is.null(post$h) || is.null(post$nu_minus2)) next
    is_regime_specific_h <- identical(model_name, "regimeSwitchingSV") &&
      length(dim(post$h)) == 3L
    T_len <- if (is_regime_specific_h) dim(post$h)[2L] else ncol(post$h)
    observed <- bench$y_all[seq_len(T_len)]
    draws <- sample.int(nrow(post$h), min(n_rep, nrow(post$h)), replace = FALSE)
    y_rep <- matrix(NA_real_, nrow = length(draws), ncol = T_len)
    is_t <- model_name != "nonlinearSV" || isTRUE(result$model$use_student_t)
    selected_regime <- if (is_regime_specific_h) {
      if (is.null(post$filtered_prob_last)) {
        rep.int(1L, length(draws))
      } else {
        ifelse(
          stats::rbinom(length(draws), 1, post$filtered_prob_last[draws, 1L]) == 1L,
          1L,
          2L
        )
      }
    } else {
      NULL
    }

    for (i in seq_along(draws)) {
      draw <- draws[i]
      h_draw <- if (is_regime_specific_h) {
        post$h[draw, , selected_regime[i]]
      } else {
        post$h[draw, ]
      }
      scale <- sqrt(exp(h_draw))
      y_rep[i, ] <- if (is_t) {
        stats::rt(T_len, df = post$nu_minus2[draw] + 2) * scale
      } else {
        stats::rnorm(T_len, sd = scale)
      }
    }

    observed_statistics <- ppc_statistics(observed, acf_lags)
    replicated_statistics <- t(apply(y_rep, 1, ppc_statistics, acf_lags = acf_lags))
    for (statistic_name in names(observed_statistics)) {
      values <- replicated_statistics[, statistic_name]
      all_rows[[counter]] <- data.frame(
        Model = model_name,
        statistic = statistic_name,
        observed = observed_statistics[[statistic_name]],
        predictive_mean = mean(values),
        predictive_p05 = as.numeric(stats::quantile(values, 0.05)),
        predictive_p95 = as.numeric(stats::quantile(values, 0.95)),
        posterior_predictive_p_upper = mean(values >= observed_statistics[[statistic_name]]),
        stringsAsFactors = FALSE
      )
      counter <- counter + 1L
    }
  }
  if (length(all_rows) == 0L) return(data.frame())
  do.call(rbind, all_rows)
}

stan_convergence_table <- function(benchmark_results) {
  stan_models <- intersect(c("nonlinearSV", "linearSV", "regimeSwitchingSV"),
                           names(benchmark_results))
  rows <- list()
  counter <- 1L
  for (model_name in stan_models) {
    diagnostics <- benchmark_results[[model_name]]$diagnostics %||% list()
    refits <- diagnostics$refits %||% c(
      diagnostics$validation_refits %||% list(),
      diagnostics$test_refits %||% list()
    )
    for (refit in refits) {
      d <- refit$diagnostics %||% list()
      rows[[counter]] <- data.frame(
        Model = model_name,
        refit_id = refit$refit_id %||% NA_integer_,
        train_T = refit$train_T %||% NA_integer_,
        max_Rhat = d$max_rhat %||% NA_real_,
        min_bulk_ESS = d$min_bulk_ess %||% NA_real_,
        min_tail_ESS = d$min_tail_ess %||% NA_real_,
        divergences = d$divergence_sum %||% NA_real_,
        min_E_BFMI = d$min_ebfmi %||% NA_real_,
        stringsAsFactors = FALSE
      )
      counter <- counter + 1L
    }
  }
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}

architecture_selection_table <- function(benchmark_results) {
  if (is.null(benchmark_results$nonlinearSV$architecture_selection)) {
    return(data.frame())
  }
  selection <- benchmark_results$nonlinearSV$architecture_selection
  candidates <- selection$candidates
  # validation_divergences/validation_clean were added 2026-08-26; older
  # cached benchmark_results won't have them, so fall back to NA rather than
  # erroring on a missing column.
  n <- length(candidates)
  divergences <- selection$validation_divergences %||% rep(NA_real_, n)
  clean <- selection$validation_clean %||% rep(NA, n)
  data.frame(
    candidate = vapply(candidates, function(x) x$label %||% paste0("K", x$K), character(1)),
    K = vapply(candidates, `[[`, integer(1), "K"),
    distribution = ifelse(
      vapply(candidates, `[[`, logical(1), "use_student_t"), "Student-t", "Gaussian"
    ),
    validation_score = selection$validation_scores,
    validation_divergences = divergences,
    validation_clean = clean,
    selected = seq_along(candidates) == selection$selected_index,
    metric = selection$metric,
    stringsAsFactors = FALSE
  )
}

# 2026-08-26 bug fix: same units mismatch as economic_results_pipeline.R's
# portfolio_metrics() (see its comment) -- actual_return/forecast_variance
# are on a PERCENT scale, but target_weekly_vol was a decimal fraction, so
# weights came out ~100x too small and this "15%-vol-targeting" portfolio
# actually ran at ~0.17% annual vol regardless of the model. The *100 below
# puts target_weekly_vol on the same percent scale as sqrt(forecast_variance)
# before the division.
volatility_targeting <- function(actual_return, forecast_variance,
                                 target_annual_vol = 0.15,
                                 max_leverage = 2) {
  target_weekly_vol <- 100 * target_annual_vol / sqrt(52)
  weights <- pmin(target_weekly_vol / sqrt(pmax(forecast_variance, .Machine$double.eps)),
                  max_leverage)
  portfolio_return <- weights * actual_return / 100
  wealth <- cumprod(1 + portfolio_return)
  drawdown <- 1 - wealth / cummax(wealth)
  annual_return <- mean(portfolio_return) * 52
  annual_volatility <- stats::sd(portfolio_return) * sqrt(52)
  list(
    annual_return = annual_return,
    annual_volatility = annual_volatility,
    sharpe = ifelse(annual_volatility > 0, annual_return / annual_volatility, NA_real_),
    max_drawdown = max(drawdown, na.rm = TRUE),
    CER = annual_return - 0.5 * 3 * annual_volatility^2
  )
}

portfolio_table <- function(benchmark_results) {
  do.call(rbind, lapply(available_forecast_models(benchmark_results), function(model_name) {
    forecast <- model_test_forecasts(benchmark_results[[model_name]], model_name)
    metrics <- volatility_targeting(forecast$actual_return, forecast$variance)
    data.frame(Model = model_name, as.data.frame(metrics), row.names = NULL)
  }))
}

# One QLIKE-per-model row, reused by both overall_evaluation_table() (the
# lightweight "just the summary" entry point) and evaluate_all_models() (the
# full evaluation) so the two don't drift out of sync with each other.
#
# 2026-08-24 bug fix: this used to call benchmark_utils.R's qlike(actual_variance,
# forecast_variance) with forecast$actual_return (unsquared) as the first
# argument. That function does NOT square its input -- it expects the caller
# to already have a variance-scale proxy -- so raw signed returns were passed
# through pmax(., eps) untouched on up days (hugely overstating the "true"
# variance) and floored to ~0 on down days (hugely understating it). This
# silently produced a QLIKE ranking that contradicted dm_QLIKE.csv's pairwise
# comparisons on all 6 model pairs (3 of them significant). qlike_loss()
# (defined above, used by loss_matrix()/the DM tests) squares actual_return
# correctly per Patton (2011); reusing it here instead of the other qlike()
# guarantees this table and the DM tests can never drift apart again.
qlike_table_for <- function(benchmark_results) {
  rows <- lapply(available_forecast_models(benchmark_results), function(name) {
    forecast <- model_test_forecasts(benchmark_results[[name]], name)
    data.frame(Model = name,
               QLIKE = mean(qlike_loss(forecast$actual_return, forecast$variance),
                            na.rm = TRUE))
  })
  do.call(rbind, rows)
}

# density/tail/portfolio/qlike_table are already per-model tables; this just
# outer-joins them on Model into the one-row-per-model summary both entry
# points return.
merge_evaluation_summary <- function(density, qlike_table, tail, portfolio) {
  tail_05 <- tail[
    tail$alpha == 0.05,
    c("Model", "FZ0_mean", "p_UC", "p_CC", "p_DQ"),
    drop = FALSE
  ]
  Reduce(
    function(x, y) merge(x, y, by = "Model", all = TRUE),
    list(density, qlike_table, tail_05, portfolio)
  )
}

overall_evaluation_table <- function(bench, benchmark_results,
                                     cpu_variable = "GCPU_baseline") {
  density <- density_score_table(bench, benchmark_results, cpu_variable)
  tail <- tail_backtest_table(benchmark_results)
  portfolio <- portfolio_table(benchmark_results)
  qlike_table <- qlike_table_for(benchmark_results)
  merge_evaluation_summary(density, qlike_table, tail, portfolio)
}

evaluate_all_models <- function(bench,
                                benchmark_results,
                                cpu_variable = "GCPU_baseline",
                                mcs_bootstrap = 5000L,
                                ppc_replications = 500L) {
  if (!length(available_forecast_models(benchmark_results))) {
    stop("benchmark_results contains no rolling forecast tables.")
  }

  density <- density_score_table(bench, benchmark_results, cpu_variable)
  tail <- tail_backtest_table(benchmark_results)
  portfolio <- portfolio_table(benchmark_results)
  qlike_table <- qlike_table_for(benchmark_results)
  summary <- merge_evaluation_summary(density, qlike_table, tail, portfolio)
  dm_tests <- list(
    QLIKE = pairwise_dm_tests(benchmark_results, "qlike"),
    negative_LPS = pairwise_dm_tests(benchmark_results, "negative_lps"),
    FZ0_5pct = pairwise_dm_tests(benchmark_results, "fz05")
  )
  loo_comparison <- bayesian_loo_comparison(benchmark_results)

  list(
    summary = summary,
    # LOO_comparison is the flat, writable view of LOO (see run_models.R's
    # table_files); LOO keeps the full loo/stacking objects for interactive use.
    LOO_comparison = loo_comparison$comparison_table,
    LOO_pareto_k = loo_comparison$pareto_k,
    architecture_selection = architecture_selection_table(benchmark_results),
    convergence = stan_convergence_table(benchmark_results),
    density_scores = density,
    tail_backtests = tail,
    dm_tests = dm_tests,
    MCS_QLIKE = model_confidence_set(benchmark_results, "qlike", B = mcs_bootstrap),
    LOO = loo_comparison,
    posterior_predictive_checks = posterior_predictive_checks(
      bench, benchmark_results, n_rep = ppc_replications),
    portfolio = portfolio
  )
}

if (isTRUE(getOption("benchmark.evaluate_autorun", TRUE)) &&
    exists("bench") && exists("benchmark_results")) {
  evaluation_results <- evaluate_all_models(bench, benchmark_results)
  summary_metrics <- evaluation_results$summary
}
