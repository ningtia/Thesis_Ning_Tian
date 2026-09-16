# Select models and settings here.

# MODELS_TO_RUN <- c("nonlinearSV")
MODELS_TO_RUN <- c("sGARCH", "gjrGARCH", "eGARCH", "linearSV","harRV","regimeSwitchingSV", "nonlinearSV")
FORCE_RERUN <- F
LOAD_SAVED_RESULTS <- T
RUN_EVALUATION <- T
SAVE_FINAL_STAN_PLOTS <- TRUE

CHECK_CONVERGENCE       <- T   # 要不要真的算 rhat/ESS/divergence
PREFLIGHT_ONLY          <- F   # 只跑训练窗口,还是滚完 validation/test
STOP_ON_NON_CONVERGENCE <- F   # 真发现没收敛时,是停下来还是继续跑

REFIT_EVERY <- 13L
SAVE_ALL_ROLLING_FITS <- FALSE
GLOBAL_SEED <- 6666L

STAN_CHAINS <- 4L
STAN_WARMUP <- 2000L
STAN_SAMPLING <- 4000L
STAN_ADAPT_DELTA <- 0.995
LINEAR_STAN_ADAPT_DELTA <- 0.99
MS_STAN_ADAPT_DELTA <- 0.995
STAN_MAX_TREEDEPTH <- 12L

NN_ARCHITECTURE_METRIC <- "qlike"
NN_COMPUTE_LOO <- TRUE
NN_S_FIXED <- 1.0
NN_TAU_W_SCALE <- 0.2
NN_STAN_SAMPLING <- 4000L

CPU_VARIABLE <- "GCPU_baseline"
MCS_BOOTSTRAP <- 5000L
PPC_REPLICATIONS <- 500L

first_file <- function(x, required = FALSE) {
  hit <- x[file.exists(x)]
  if (length(hit)) hit[1L] else NULL
}

check_result <- function(x) {
  needed <- c(
    "forecast_date", "split", "actual_return", "variance",
    "VaR_01", "ES_01", "VaR_05", "ES_05", "log_score"
  )
  is.list(x) && is.data.frame(x$forecast_table) &&
    nrow(x$forecast_table) > 0L && all(needed %in% names(x$forecast_table))
}

model_path <- function(key, model_dir) file.path(model_dir, paste0(key, ".rds"))

persist_result_index <- function(results, model_dir) {
  saveRDS(results, file.path(model_dir, "benchmark_results_all_available.rds"),
          compress = TRUE)
  results
}

persist_model_result <- function(key, result, model_dir, results) {
  if (!check_result(result)) stop(key, " returned an invalid forecast_table.", call. = FALSE)
  saveRDS(result, model_path(key, model_dir), compress = TRUE)
  results[[key]] <- result
  persist_result_index(results, model_dir)
}

load_saved_results <- function(keys, model_dir) {
  results <- list()
  for (key in keys) {
    path <- model_path(key, model_dir)
    if (file.exists(path)) {
      result <- tryCatch(readRDS(path), error = function(e) NULL)
      if (check_result(result)) results[[key]] <- result
    }
  }
  results
}

run_model_sequence <- function(runners, model_dir, initial_results = list()) {
  results <- initial_results
  errors <- list()
  for (key in names(runners)) {
    outcome <- tryCatch({
      result <- runners[[key]]()
      if (inherits(result, "benchmark_preflight_failure")) {
        stop(result$failure_reason, call. = FALSE)
      }
      list(ok = TRUE, results = persist_model_result(key, result, model_dir, results))
    }, error = function(e) list(ok = FALSE, error = conditionMessage(e)))

    if (isTRUE(outcome$ok)) {
      results <- outcome$results
    } else {
      errors[[key]] <- outcome$error
      results[[key]] <- NULL
      persist_result_index(results, model_dir)
      warning(key, " failed: ", outcome$error, call. = FALSE)
    }
  }
  list(benchmark_results = results, errors = errors)
}

run_preflight_sequence <- function(runners, diagnostics_dir) {
  reports <- list()
  errors <- list()
  for (key in names(runners)) {
    report <- tryCatch(runners[[key]](), error = function(e) e)

    if (inherits(report, "error")) {
      errors[[key]] <- conditionMessage(report)
      warning(key, " preflight failed: ", errors[[key]], call. = FALSE)
    } else {
      if (!is.null(report)) reports[[key]] <- report
      if (is.null(report) || !isTRUE(report$passed)) {
        message_text <- if (is.null(report) || is.null(report$message)) {
          "Preflight failed."
        } else {
          report$message
        }
        errors[[key]] <- message_text
        warning(key, " preflight failed: ", message_text, call. = FALSE)
      }
    }
  }

  report_path <- file.path(diagnostics_dir, "preflight_reports.rds")
  saved_reports <- if (file.exists(report_path)) {
    tryCatch(readRDS(report_path), error = function(e) list())
  } else list()
  compact_reports <- lapply(reports, function(x) {
    x$fit <- NULL
    x
  })
  saved_reports[names(compact_reports)] <- compact_reports
  saveRDS(saved_reports, report_path, compress = TRUE)
  saveRDS(errors, file.path(diagnostics_dir, "preflight_errors.rds"), compress = TRUE)
  list(reports = reports, errors = errors)
}

resolve_selection <- function(x, all_keys) {
  if ("all" %in% x) all_keys else intersect(x, all_keys)
}

save_final_stan_plot <- function(results, key, diagnostics_dir, pars, pairs) {
  result <- results[[key]]
  if (!is.null(result$fit) && inherits(result$fit, "stanfit") &&
      exists("save_stan_diagnostic_plots", mode = "function")) {
    save_stan_diagnostic_plots(
      result$fit,
      file.path(diagnostics_dir, paste0(key, "_final_refit.pdf")),
      pars,
      pairs
    )
  }
}

run_garch_one <- function(key, bench) {
  run_garch_benchmark(
    key,
    bench,
    REFIT_EVERY,
    SAVE_ALL_ROLLING_FITS,
    GLOBAL_SEED,
    solver = "hybrid",
    fallback_solver = if (identical(key, "eGARCH")) "solnp" else NULL
  )
}

run_linear_one <- function(bench) {
  run_linear_sv(
    bench,
    REFIT_EVERY,
    SAVE_ALL_ROLLING_FITS,
    GLOBAL_SEED,
    STAN_CHAINS,
    STAN_WARMUP + STAN_SAMPLING,
    STAN_WARMUP,
    LINEAR_STAN_ADAPT_DELTA,
    STAN_MAX_TREEDEPTH
  )
}

run_regime_switching_one <- function(bench) {
  run_regime_switching_sv(
    bench,
    REFIT_EVERY,
    SAVE_ALL_ROLLING_FITS,
    GLOBAL_SEED,
    STAN_CHAINS,
    STAN_WARMUP + STAN_SAMPLING,
    STAN_WARMUP,
    MS_STAN_ADAPT_DELTA,
    STAN_MAX_TREEDEPTH
  )
}

run_nonlinear_one <- function(bench) {
  config <- nn_sv_default_config()
  config$refit_every <- REFIT_EVERY
  config$save_fits <- SAVE_ALL_ROLLING_FITS
  config$seed <- GLOBAL_SEED
  config$chains <- STAN_CHAINS
  config$warmup <- STAN_WARMUP
  config$iter <- STAN_WARMUP + NN_STAN_SAMPLING
  config$adapt_delta <- STAN_ADAPT_DELTA
  config$max_treedepth <- STAN_MAX_TREEDEPTH
  config$compute_loo <- NN_COMPUTE_LOO
  run_nonlinear_sv(
    bench,
    default_nn_candidates(),
    NN_ARCHITECTURE_METRIC,
    config
  )
}

run_har_one <- function(bench) {
  run_har_rv(bench, REFIT_EVERY, SAVE_ALL_ROLLING_FITS, GLOBAL_SEED)
}

run_stan_preflight_only <- function(model_name, stan_file, preflight_function,
                                    adapt_delta, bench) {
  cores <- parallel::detectCores(logical = TRUE)
  if (is.na(cores) || cores < 1L) cores <- 1L
  options(mc.cores = min(STAN_CHAINS, cores))
  rstan::rstan_options(auto_write = TRUE)
  compiled_model <- rstan::stan_model(file = stan_file)
  preflight <- preflight_function(
    compiled_model,
    bench,
    STAN_CHAINS,
    STAN_WARMUP + STAN_SAMPLING,
    STAN_WARMUP,
    GLOBAL_SEED,
    adapt_delta,
    STAN_MAX_TREEDEPTH
  )
  list(
    model_name = model_name,
    kind = "stan",
    passed = isTRUE(preflight$passed),
    message = preflight$message,
    diagnostics = preflight$diagnostics,
    train_indices = preflight$train_indices,
    candidate = NULL,
    sampler_config = list(
      chains = STAN_CHAINS,
      iter = STAN_WARMUP + STAN_SAMPLING,
      warmup = STAN_WARMUP,
      adapt_delta = adapt_delta,
      max_treedepth = STAN_MAX_TREEDEPTH
    ),
    parallel = list(available_cores = cores,
                    concurrent_chains = min(STAN_CHAINS, cores)),
    fit = preflight$fitted$fit
  )
}

run_linear_preflight_only <- function(bench) {
  run_stan_preflight_only(
    "linearSV", "linear_sv.stan", run_linear_sv_preflight,
    LINEAR_STAN_ADAPT_DELTA, bench
  )
}

run_regime_switching_preflight_only <- function(bench) {
  run_stan_preflight_only(
    "regimeSwitchingSV", "regime_switching_sv.stan",
    run_regime_switching_sv_preflight, MS_STAN_ADAPT_DELTA, bench
  )
}

run_nonlinear_preflight_only <- function(bench) {
  config <- nn_sv_default_config()
  config$refit_every <- REFIT_EVERY
  config$save_fits <- FALSE
  config$seed <- GLOBAL_SEED
  config$chains <- STAN_CHAINS
  config$warmup <- STAN_WARMUP
  config$iter <- STAN_WARMUP + NN_STAN_SAMPLING
  config$adapt_delta <- STAN_ADAPT_DELTA
  config$max_treedepth <- STAN_MAX_TREEDEPTH
  config$compute_loo <- FALSE
  parallel_config <- configure_nn_sv_parallel(config$chains)
  compiled_model <- rstan::stan_model(file = config$stan_file)
  preflight <- run_nn_sv_preflight(
    compiled_model,
    bench,
    default_nn_candidates()[[1L]],
    config
  )
  list(
    model_name = "nonlinearSV",
    kind = "stan",
    passed = isTRUE(preflight$passed),
    message = preflight$message,
    diagnostics = preflight$diagnostics,
    train_indices = preflight$train_indices,
    candidate = preflight$candidate,
    sampler_config = config,
    parallel = parallel_config,
    fit = preflight$fitted$fit
  )
}

run_har_preflight_only <- function(bench) {
  prepared <- prepare_har_rv_data(bench)
  forecast_indices <- sort(c(bench$split_rows$valid, bench$split_rows$test))
  train_indices <- seq_len(min(forecast_indices) - 1L)
  fitted <- make_har_rv_fitter(prepared$data, prepared$formula)(
    train_indices,
    NULL,
    1L,
    bench
  )
  coefficients <- stats::coef(fitted$fit)
  passed <- stats::nobs(fitted$fit) > length(coefficients) &&
    all(is.finite(coefficients)) &&
    is.finite(fitted$model$sigma2_log) && fitted$model$sigma2_log >= 0
  list(
    model_name = "harRV",
    kind = "ols",
    passed = passed,
    message = if (passed) "Initial HAR-RV OLS data and fit preflight passed."
              else "Initial HAR-RV OLS data and fit preflight failed.",
    diagnostics = fitted$diagnostics,
    train_indices = train_indices,
    fit = fitted$fit
  )
}

stan_plot_requests <- function() {
  list(
    linearSV = list(
      pars = c("mu", "phi", "phi_star", "sigma_eta", "nu_minus2"),
      pairs = c("phi", "sigma_eta", "nu_minus2")
    ),
    regimeSwitchingSV = list(
      pars = c("mu", "phi_unit", "log_sigma_eta", "p11_unit", "p22_unit", "nu_minus2"),
      pairs = c("phi_unit", "log_sigma_eta", "p11_unit", "p22_unit")
    ),
    nonlinearSV = list(
      pars = c("mu", "phi", "phi_star", "sigma_eta", "tau_w"),
      pairs = c("phi", "sigma_eta", "tau_w")
    )
  )
}

save_preflight_stan_plots <- function(reports, diagnostics_dir) {
  requests <- stan_plot_requests()
  for (key in intersect(names(requests), names(reports))) {
    report <- reports[[key]]
    if (inherits(report$fit, "stanfit")) {
      tryCatch(
        save_stan_diagnostic_plots(
          report$fit,
          file.path(diagnostics_dir, paste0(key, "_preflight_diagnostics.pdf")),
          requests[[key]]$pars,
          requests[[key]]$pairs
        ),
        error = function(e) warning(conditionMessage(e), call. = FALSE)
      )
    }
  }
}

run_benchmark_suite <- function() {
  files <- list(
    utils = first_file("benchmark_utils.R"),
    garch = first_file("GARCH.R"),
    linear = first_file("linear_sv.R"),
    ms = first_file("regime_switching_sv.R"),
    nn = first_file("nonlinear_sv.R"),
    har = first_file("HAR_RV.R"),
    evaluation = first_file("evaluate_results.R")
  )
  all_keys <- c(
    "sGARCH", "gjrGARCH", "eGARCH", "linearSV",
    "regimeSwitchingSV", "nonlinearSV", "harRV"
  )
  selected <- resolve_selection(MODELS_TO_RUN, all_keys)

  options(
    benchmark.garch_autorun = FALSE,
    benchmark.linear_autorun = FALSE,
    benchmark.linear_sv_autorun = FALSE,
    benchmark.ms_autorun = FALSE,
    benchmark.ms_sv_autorun = FALSE,
    benchmark.nn_autorun = FALSE,
    benchmark.har_autorun = FALSE,
    benchmark.evaluate_autorun = FALSE,
    benchmark.refit_every = REFIT_EVERY,
    benchmark.save_fits = SAVE_ALL_ROLLING_FITS,
    benchmark.stan_chains = STAN_CHAINS,
    benchmark.stan_warmup = STAN_WARMUP,
    benchmark.stan_iter = STAN_WARMUP + STAN_SAMPLING,
    benchmark.stan_adapt_delta = STAN_ADAPT_DELTA,
    benchmark.linear_stan_adapt_delta = LINEAR_STAN_ADAPT_DELTA,
    benchmark.ms_stan_adapt_delta = MS_STAN_ADAPT_DELTA,
    benchmark.stan_max_treedepth = STAN_MAX_TREEDEPTH,
    benchmark.nn_chains = STAN_CHAINS,
    benchmark.nn_warmup = STAN_WARMUP,
    benchmark.nn_sampling = STAN_SAMPLING,
    benchmark.nn_iter = STAN_WARMUP + STAN_SAMPLING,
    benchmark.nn_adapt_delta = STAN_ADAPT_DELTA,
    benchmark.nn_max_treedepth = STAN_MAX_TREEDEPTH,
    benchmark.nn_compute_loo = NN_COMPUTE_LOO,
    benchmark.check_convergence = CHECK_CONVERGENCE,
    benchmark.stop_on_non_convergence = STOP_ON_NON_CONVERGENCE,
    benchmark.architecture_metric = NN_ARCHITECTURE_METRIC,
    benchmark.nn_s_fixed = NN_S_FIXED,
    benchmark.nn_tau_w_scale = NN_TAU_W_SCALE,
    benchmark.nn_seed = GLOBAL_SEED
  )
  set.seed(GLOBAL_SEED)

  if (!is.null(files$utils)) source(files$utils)
  module_files <- list(
    sGARCH = files$garch,
    gjrGARCH = files$garch,
    eGARCH = files$garch,
    linearSV = files$linear,
    regimeSwitchingSV = files$ms,
    nonlinearSV = files$nn,
    harRV = files$har
  )
  source_errors <- list()
  for (file in unique(unlist(module_files[selected], use.names = FALSE))) {
    error_message <- tryCatch({
      source(file)
      NULL
    }, error = function(e) conditionMessage(e))
    if (!is.null(error_message)) source_errors[[file]] <- error_message
  }

  bench <- load_benchmark_data()
  directories <- list(
    models = "results/models",
    evaluation = "results/evaluation",
    tables = "results/tables",
    diagnostics = "results/diagnostics"
  )
  invisible(lapply(directories, dir.create, recursive = TRUE, showWarnings = FALSE))
  results <- if (isTRUE(LOAD_SAVED_RESULTS)) {
    load_saved_results(all_keys, directories$models)
  } else list()
  if (!isTRUE(FORCE_RERUN)) {
    cached_results <- load_saved_results(selected, directories$models)
    results[names(cached_results)] <- cached_results
  }

  registry <- list(
    sGARCH = function() run_garch_one("sGARCH", bench),
    gjrGARCH = function() run_garch_one("gjrGARCH", bench),
    eGARCH = function() run_garch_one("eGARCH", bench),
    linearSV = function() run_linear_one(bench),
    regimeSwitchingSV = function() run_regime_switching_one(bench),
    nonlinearSV = function() run_nonlinear_one(bench),
    harRV = function() run_har_one(bench)
  )

  if (isTRUE(PREFLIGHT_ONLY)) {
    preflight_registry <- list(
      linearSV = function() run_linear_preflight_only(bench),
      regimeSwitchingSV = function() run_regime_switching_preflight_only(bench),
      nonlinearSV = function() run_nonlinear_preflight_only(bench),
      harRV = function() run_har_preflight_only(bench)
    )
    preflight_keys <- intersect(selected, names(preflight_registry))
    preflight_keys <- preflight_keys[vapply(preflight_keys, function(key) {
      file <- module_files[[key]]
      !is.null(file) && !(file %in% names(source_errors))
    }, logical(1))]
    preflight_runners <- preflight_registry[preflight_keys]
    preflight_result <- run_preflight_sequence(preflight_runners, directories$diagnostics)
    failed_sources <- setdiff(intersect(selected, names(preflight_registry)), preflight_keys)
    for (key in failed_sources) {
      file <- module_files[[key]]
      if (!is.null(file) && file %in% names(source_errors)) {
        preflight_result$errors[[key]] <- source_errors[[file]]
      }
    }
    unavailable <- setdiff(selected, names(preflight_registry))
    preflight_result$errors[unavailable] <- "Preflight-only mode is not available for this model."
    saveRDS(preflight_result$errors,
            file.path(directories$diagnostics, "preflight_errors.rds"),
            compress = TRUE)
    if (isTRUE(SAVE_FINAL_STAN_PLOTS)) {
      save_preflight_stan_plots(preflight_result$reports, directories$diagnostics)
    }
    return(invisible(preflight_result))
  }

  errors <- list()
  runnable <- list()
  for (key in selected) {
    file <- module_files[[key]]
    if (is.null(file)) next
    if (file %in% names(source_errors)) {
      errors[[key]] <- source_errors[[file]]
      results[[key]] <- NULL
      next
    }
    if (!isTRUE(FORCE_RERUN) && key %in% names(results)) next
    runnable[[key]] <- registry[[key]]
  }

  sequence_result <- run_model_sequence(runnable, directories$models, results)
  results <- sequence_result$benchmark_results
  errors[names(sequence_result$errors)] <- sequence_result$errors

  if (isTRUE(LOAD_SAVED_RESULTS)) {
    disk_results <- load_saved_results(all_keys, directories$models)
    disk_results[names(errors)] <- NULL
    results[names(disk_results)] <- disk_results
  }
  persist_result_index(results, directories$models)
  saveRDS(errors, file.path(directories$models, "benchmark_run_errors.rds"),
          compress = TRUE)

  if (isTRUE(SAVE_FINAL_STAN_PLOTS)) {
    requests <- stan_plot_requests()
    for (key in names(requests)) {
      tryCatch(
        save_final_stan_plot(
          results,
          key,
          directories$diagnostics,
          requests[[key]]$pars,
          requests[[key]]$pairs
        ),
        error = function(e) warning(conditionMessage(e), call. = FALSE)
      )
    }
  }

  evaluation_results <- NULL
  if (isTRUE(RUN_EVALUATION) && !is.null(files$evaluation) && length(results)) {
    evaluation_results <- tryCatch({
      source(files$evaluation)
      output <- evaluate_all_models(
        bench,
        results,
        CPU_VARIABLE,
        MCS_BOOTSTRAP,
        PPC_REPLICATIONS
      )
      saveRDS(
        output,
        file.path(directories$evaluation, "evaluation_results_all_available.rds"),
        compress = TRUE
      )
      table_files <- c(
        summary = "overall_summary.csv",
        architecture_selection = "architecture_selection.csv",
        convergence = "stan_convergence.csv",
        density_scores = "density_scores.csv",
        LOO_comparison = "loo_comparison.csv",
        LOO_pareto_k = "loo_pareto_k.csv",
        tail_backtests = "tail_backtests.csv",
        portfolio = "portfolio_results.csv",
        posterior_predictive_checks = "posterior_predictive_checks.csv"
      )
      for (item in names(table_files)) {
        if (is.data.frame(output[[item]])) {
          utils::write.csv(
            output[[item]],
            file.path(directories$tables, table_files[[item]]),
            row.names = FALSE
          )
        }
      }
      if (is.list(output$dm_tests)) {
        dm_files <- c(
          QLIKE = "dm_qlike.csv",
          negative_LPS = "dm_negative_lps.csv",
          FZ0_5pct = "dm_fz05.csv"
        )
        for (item in intersect(names(output$dm_tests), names(dm_files))) {
          if (is.data.frame(output$dm_tests[[item]])) {
            utils::write.csv(
              output$dm_tests[[item]],
              file.path(directories$tables, dm_files[[item]]),
              row.names = FALSE
            )
          }
        }
      }
      # 2026-08-26: evaluate_all_models() already computes MCS_QLIKE (via
      # MCS::MCSprocedure(), when the MCS package -- not part of Rlib/, only
      # on machines with it installed separately -- is available), but until
      # now nothing wrote it out: it was reachable only by loading
      # evaluation_results_all_available.rds and printing the S4 object by
      # hand. writeout is wrapped in tryCatch because obj@show's presence and
      # shape depend on MCS's internal SSM class, which this project doesn't
      # control.
      if (isTRUE(output$MCS_QLIKE$available)) {
        mcs_written <- tryCatch({
          obj <- output$MCS_QLIKE$object
          m <- obj@show
          mcs_table <- data.frame(
            Model = rownames(m),
            Avg_Loss = m[, "Avg.Loss"],
            p_Value_H0 = m[, "p-Value for H_{0,M_k}"],
            MCS_p_Value = m[, "MCS p-Value"],
            row.names = NULL
          )
          mcs_table$in_MCS <- mcs_table$MCS_p_Value > obj@Info$alpha
          mcs_table <- mcs_table[order(-mcs_table$MCS_p_Value), ]
          utils::write.csv(
            mcs_table,
            file.path(directories$tables, "mcs_qlike.csv"),
            row.names = FALSE
          )
          utils::write.csv(
            data.frame(
              field = c("loss", "alpha", "statistic", "B_bootstrap", "seed",
                        "n_eliminated", "n_total_models", "elapsed_seconds"),
              value = c("qlike", obj@Info$alpha, obj@Info$statistic, obj@Info$B,
                        obj@Info$seed, obj@Info$n_elim,
                        length(obj@Info$model.names),
                        round(as.numeric(obj@Info$elapsed.time), 3))
            ),
            file.path(directories$tables, "mcs_qlike_meta.csv"),
            row.names = FALSE
          )
          TRUE
        }, error = function(e) {
          warning("Could not write mcs_qlike.csv: ", conditionMessage(e), call. = FALSE)
          FALSE
        })
      }
      output
    }, error = function(e) {
      warning("Evaluation failed: ", conditionMessage(e), call. = FALSE)
      NULL
    })
  }

  list(
    benchmark_results = results,
    errors = errors,
    evaluation_results = evaluation_results
  )
}

if (isTRUE(getOption("benchmark.runner_autorun", TRUE))) {
  suite_result <- run_benchmark_suite()
  benchmark_results <- suite_result$benchmark_results
  evaluation_results <- suite_result$evaluation_results
}
