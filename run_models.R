################################################################################
# run_models.R
# Run one model, several models, all models, or evaluation only.
################################################################################

# ========================== 1. EDIT ONLY HERE ================================

# "all"; one model; several models; or character(0) for evaluation only.
MODELS_TO_RUN <- "all"
# MODELS_TO_RUN <- "sGARCH"
# MODELS_TO_RUN <- c("sGARCH", "gjrGARCH", "eGARCH")
# MODELS_TO_RUN <- "nonlinearSV"
# MODELS_TO_RUN <- c("linearSV", "nonlinearSV")
# MODELS_TO_RUN <- "harRV"
# MODELS_TO_RUN <- "regimeSwitchingSV"
# MODELS_TO_RUN <- character(0)

FORCE_RERUN <- FALSE          # TRUE = overwrite selected saved models
LOAD_SAVED_RESULTS <- TRUE    # include models completed in earlier sessions
RUN_EVALUATION <- TRUE
SAVE_FINAL_STAN_PLOTS <- TRUE

REFIT_EVERY <- 13L
SAVE_ALL_ROLLING_FITS <- FALSE
GLOBAL_SEED <- 6666L

STAN_CHAINS <- 4L
STAN_WARMUP <- 1500L
STAN_SAMPLING <- 2000L
STAN_ADAPT_DELTA <- 0.95
STAN_MAX_TREEDEPTH <- 12L

NN_ARCHITECTURE_METRIC <- "qlike"
NN_COMPUTE_LOO <- TRUE

CPU_VARIABLE <- "GCPU_baseline"
MCS_BOOTSTRAP <- 5000L
PPC_REPLICATIONS <- 500L

# ========================== 2. HELPERS =======================================

first_file <- function(x, required = TRUE) {
  hit <- x[file.exists(x)]
  if (length(hit)) return(hit[1L])
  if (required) stop("File not found: ", paste(x, collapse = " / "), call. = FALSE)
  NULL
}

check_result <- function(x, key) {
  if (!is.list(x) || !is.data.frame(x$forecast_table) ||
      nrow(x$forecast_table) == 0L) {
    stop(key, " did not return a non-empty forecast_table.", call. = FALSE)
  }
  needed <- c(
    "forecast_date", "split", "actual_return", "variance", "VaR_01",
    "ES_01", "VaR_05", "ES_05", "log_score"
  )
  missing <- setdiff(needed, names(x$forecast_table))
  if (length(missing)) {
    stop(key, " forecast_table missing: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  invisible(TRUE)
}

model_path <- function(key, model_dir) {
  file.path(model_dir, paste0(key, ".rds"))
}

persist_result_index <- function(results, model_dir) {
  saveRDS(
    results,
    file.path(model_dir, "benchmark_results_all_available.rds"),
    compress = TRUE
  )
  results
}

persist_model_result <- function(key, result, model_dir, results) {
  check_result(result, key)
  saveRDS(result, model_path(key, model_dir), compress = TRUE)
  results[[key]] <- result
  persist_result_index(results, model_dir)
}

load_saved_results <- function(keys, model_dir) {
  results <- list()
  for (key in keys) {
    path <- model_path(key, model_dir)
    if (!file.exists(path)) next

    result <- tryCatch(readRDS(path), error = function(e) NULL)
    if (is.null(result)) next

    is_valid <- tryCatch({
      check_result(result, key)
      TRUE
    }, error = function(e) FALSE)
    if (is_valid) results[[key]] <- result
  }
  results
}

run_model_sequence <- function(runners, model_dir, initial_results = list()) {
  results <- initial_results
  errors <- list()

  for (key in names(runners)) {
    message("\n", strrep("=", 70), "\nRUNNING: ", key,
            "\n", strrep("=", 70))
    started <- Sys.time()

    outcome <- tryCatch(
      {
        result <- runners[[key]]()
        if (is.null(result)) {
          stop(key, " runner returned NULL.", call. = FALSE)
        }
        list(ok = TRUE, results = persist_model_result(
          key = key,
          result = result,
          model_dir = model_dir,
          results = results
        ))
      },
      error = function(e) list(ok = FALSE, error = conditionMessage(e))
    )

    if (isTRUE(outcome$ok)) {
      results <- outcome$results
      minutes <- as.numeric(difftime(Sys.time(), started, units = "mins"))
      message("Saved ", key, " to ", model_path(key, model_dir), " (",
              round(minutes, 2), " min).")
    } else {
      errors[[key]] <- outcome$error
      warning(key, " failed: ", outcome$error, call. = FALSE)
      tryCatch(
        persist_result_index(results, model_dir),
        error = function(e) warning(
          "Could not update the result index after ", key, " failed: ",
          conditionMessage(e), call. = FALSE
        )
      )
    }
  }

  list(benchmark_results = results, errors = errors)
}

resolve_selection <- function(x, all_keys) {
  if (!length(x)) return(character())
  if ("all" %in% x) return(all_keys)
  bad <- setdiff(x, all_keys)
  if (length(bad)) stop("Unknown model(s): ", paste(bad, collapse = ", "))
  unique(x)
}

save_final_stan_plot <- function(results, key, diagnostics_dir, pars, pairs) {
  result <- results[[key]]
  if (is.null(result$fit) || !inherits(result$fit, "stanfit")) {
    return(invisible(FALSE))
  }
  if (!exists("save_stan_diagnostic_plots", mode = "function")) {
    return(invisible(FALSE))
  }
  save_stan_diagnostic_plots(
    fit = result$fit,
    file = file.path(diagnostics_dir, paste0(key, "_final_refit.pdf")),
    pars = pars,
    pairs_pars = pairs
  )
  invisible(TRUE)
}

# ========================== 3. MODEL ADAPTERS ================================

run_garch_one <- function(key, bench) {
  solver <- if (identical(key, "eGARCH")) "gosolnp" else "hybrid"
  run_garch_benchmark(
    model_name = key,
    bench = bench,
    refit_every = REFIT_EVERY,
    save_fits = SAVE_ALL_ROLLING_FITS,
    seed = GLOBAL_SEED,
    solver = solver
  )
}

run_linear_one <- function(bench) {
  run_linear_sv(
    bench = bench,
    refit_every = REFIT_EVERY,
    save_fits = SAVE_ALL_ROLLING_FITS,
    seed = GLOBAL_SEED,
    chains = STAN_CHAINS,
    iter = STAN_WARMUP + STAN_SAMPLING,
    warmup = STAN_WARMUP,
    adapt_delta = STAN_ADAPT_DELTA,
    max_treedepth = STAN_MAX_TREEDEPTH
  )
}

run_regime_switching_one <- function(bench) {
  run_regime_switching_sv(
    bench = bench,
    refit_every = REFIT_EVERY,
    save_fits = SAVE_ALL_ROLLING_FITS,
    seed = GLOBAL_SEED,
    chains = STAN_CHAINS,
    iter = STAN_WARMUP + STAN_SAMPLING,
    warmup = STAN_WARMUP,
    adapt_delta = STAN_ADAPT_DELTA,
    max_treedepth = STAN_MAX_TREEDEPTH
  )
}

run_nonlinear_one <- function(bench) {
  if (!exists("run_nonlinear_sv", mode = "function")) {
    stop("run_nonlinear_sv() not found.", call. = FALSE)
  }

  config <- nn_sv_default_config()
  config$refit_every <- REFIT_EVERY
  config$save_fits <- SAVE_ALL_ROLLING_FITS
  config$seed <- GLOBAL_SEED
  config$chains <- STAN_CHAINS
  config$warmup <- STAN_WARMUP
  config$iter <- STAN_WARMUP + STAN_SAMPLING
  config$adapt_delta <- STAN_ADAPT_DELTA
  config$max_treedepth <- STAN_MAX_TREEDEPTH
  config$compute_loo <- NN_COMPUTE_LOO

  run_nonlinear_sv(
    bench = bench,
    candidates = default_nn_candidates(),
    architecture_metric = NN_ARCHITECTURE_METRIC,
    config = config
  )
}

run_har_one <- function(bench) {
  run_har_rv(
    bench = bench,
    refit_every = REFIT_EVERY,
    save_fits = SAVE_ALL_ROLLING_FITS,
    seed = GLOBAL_SEED
  )
}

# ========================== 4. TOP-LEVEL SUITE ===============================

run_benchmark_suite <- function() {
  files <- list(
    utils = first_file("benchmark_utils.R"),
    garch = first_file("GARCH.R", FALSE),
    linear = first_file("linear_sv.R", FALSE),
    ms = first_file("regime_switching_sv.R", FALSE),
    nn = first_file("nonlinear_sv.R", FALSE),
    har = first_file("HAR_RV.R", FALSE),
    evaluation = first_file("evaluate_results.R", FALSE)
  )
  all_keys <- c(
    "sGARCH", "gjrGARCH", "eGARCH", "linearSV", "regimeSwitchingSV",
    "nonlinearSV", "harRV"
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
    benchmark.stan_max_treedepth = STAN_MAX_TREEDEPTH,
    benchmark.nn_chains = STAN_CHAINS,
    benchmark.nn_warmup = STAN_WARMUP,
    benchmark.nn_sampling = STAN_SAMPLING,
    benchmark.nn_iter = STAN_WARMUP + STAN_SAMPLING,
    benchmark.nn_adapt_delta = STAN_ADAPT_DELTA,
    benchmark.nn_max_treedepth = STAN_MAX_TREEDEPTH,
    benchmark.nn_compute_loo = NN_COMPUTE_LOO,
    benchmark.architecture_metric = NN_ARCHITECTURE_METRIC,
    benchmark.nn_seed = GLOBAL_SEED
  )
  set.seed(GLOBAL_SEED)

  source(files$utils)
  modules <- c(
    garch = files$garch,
    linearSV = files$linear,
    regimeSwitchingSV = files$ms,
    nonlinearSV = files$nn,
    harRV = files$har
  )
  module_errors <- list()
  for (key in names(modules)) {
    file <- modules[[key]]
    if (is.null(file)) next
    message("Sourcing ", file)
    tryCatch(
      source(file),
      error = function(e) {
        module_errors[[key]] <<- conditionMessage(e)
        warning("Could not source ", file, ": ", conditionMessage(e),
                call. = FALSE)
      }
    )
  }

  bench <- load_benchmark_data()
  if (!all(c("all", "y_all", "X_all", "split_rows") %in% names(bench))) {
    stop("bench is missing all/y_all/X_all/split_rows.", call. = FALSE)
  }

  directories <- list(
    models = "results/models",
    evaluation = "results/evaluation",
    tables = "results/tables",
    diagnostics = "results/diagnostics"
  )
  invisible(lapply(directories, dir.create, recursive = TRUE, showWarnings = FALSE))

  results <- if (isTRUE(LOAD_SAVED_RESULTS)) {
    load_saved_results(all_keys, directories$models)
  } else {
    list()
  }

  registry <- list(
    sGARCH = list(file = files$garch, run = function() run_garch_one("sGARCH", bench)),
    gjrGARCH = list(file = files$garch, run = function() run_garch_one("gjrGARCH", bench)),
    eGARCH = list(file = files$garch, run = function() run_garch_one("eGARCH", bench)),
    linearSV = list(file = files$linear, run = function() run_linear_one(bench)),
    regimeSwitchingSV = list(file = files$ms, run = function() run_regime_switching_one(bench)),
    nonlinearSV = list(file = files$nn, run = function() run_nonlinear_one(bench)),
    harRV = list(file = files$har, run = function() run_har_one(bench))
  )

  runnable <- list()
  errors <- module_errors
  for (key in selected) {
    entry <- registry[[key]]
    if (is.null(entry$file)) {
      message("Skipping ", key, ": source file not found.")
      next
    }
    if (!isTRUE(FORCE_RERUN) && key %in% names(results)) {
      message("Skipping ", key, ": valid saved result exists. ",
              "Set FORCE_RERUN <- TRUE to rerun.")
      next
    }
    if (key %in% names(module_errors)) {
      errors[[key]] <- module_errors[[key]]
      tryCatch(
        persist_result_index(results, directories$models),
        error = function(e) warning(
          "Could not update the result index after ", key, " failed to source: ",
          conditionMessage(e), call. = FALSE
        )
      )
      next
    }
    runnable[[key]] <- entry$run
  }

  sequence_result <- run_model_sequence(
    runners = runnable,
    model_dir = directories$models,
    initial_results = results
  )
  results <- sequence_result$benchmark_results
  errors[names(sequence_result$errors)] <- sequence_result$errors

  if (isTRUE(LOAD_SAVED_RESULTS)) {
    disk_results <- load_saved_results(all_keys, directories$models)
    results[names(disk_results)] <- disk_results
  }
  persist_result_index(results, directories$models)
  saveRDS(errors, file.path(directories$models, "benchmark_run_errors.rds"),
          compress = TRUE)

  if (isTRUE(SAVE_FINAL_STAN_PLOTS)) {
    plot_requests <- list(
      linearSV = list(
        pars = c("mu", "phi", "phi_raw", "sigma_eta", "nu_minus2"),
        pairs = c("phi", "phi_raw", "sigma_eta", "nu_minus2")
      ),
      regimeSwitchingSV = list(
        pars = c("mu", "phi", "sigma_eta", "p11", "p22", "nu_minus2"),
        pairs = c("phi", "sigma_eta", "p11", "p22")
      ),
      nonlinearSV = list(
        pars = c("mu", "phi", "phi_raw", "sigma_eta", "tau_w", "s"),
        pairs = c("phi", "phi_raw", "sigma_eta", "tau_w", "s")
      )
    )
    for (key in names(plot_requests)) {
      tryCatch(
        save_final_stan_plot(
          results = results,
          key = key,
          diagnostics_dir = directories$diagnostics,
          pars = plot_requests[[key]]$pars,
          pairs = plot_requests[[key]]$pairs
        ),
        error = function(e) warning(
          "Could not save final Stan plot for ", key, ": ",
          conditionMessage(e), call. = FALSE
        )
      )
    }
  }

  evaluation_results <- NULL
  if (isTRUE(RUN_EVALUATION) && length(results)) {
    if (is.null(files$evaluation)) {
      warning("Evaluation file not found.", call. = FALSE)
    } else {
      evaluation_results <- tryCatch(
        {
          source(files$evaluation)
          output <- evaluate_all_models(
            bench = bench,
            benchmark_results = results,
            cpu_variable = CPU_VARIABLE,
            mcs_bootstrap = MCS_BOOTSTRAP,
            ppc_replications = PPC_REPLICATIONS
          )
          saveRDS(
            output,
            file.path(directories$evaluation,
                      "evaluation_results_all_available.rds"),
            compress = TRUE
          )
          for (item in c(
            "summary", "architecture_selection", "convergence", "density_scores",
            "tail_backtests", "portfolio", "posterior_predictive_checks"
          )) {
            if (is.data.frame(output[[item]])) {
              utils::write.csv(
                output[[item]],
                file.path(directories$tables, paste0(item, ".csv")),
                row.names = FALSE
              )
            }
          }
          if (is.list(output$dm_tests)) {
            for (item in names(output$dm_tests)) {
              if (is.data.frame(output$dm_tests[[item]])) {
                utils::write.csv(
                  output$dm_tests[[item]],
                  file.path(directories$tables, paste0("dm_", item, ".csv")),
                  row.names = FALSE
                )
              }
            }
          }
          output
        },
        error = function(e) {
          warning("Evaluation failed: ", conditionMessage(e), call. = FALSE)
          NULL
        }
      )
    }
  }

  cat("\n", strrep("=", 70), "\nRUN COMPLETE\n", strrep("=", 70), "\n",
      sep = "")
  cat("Available results: ",
      if (length(results)) paste(names(results), collapse = ", ") else "<none>",
      "\n", sep = "")
  if (length(errors)) {
    cat("Failures:\n")
    for (key in names(errors)) cat("  - ", key, ": ", errors[[key]], "\n", sep = "")
  }
  cat("\nModel RDS: ", directories$models,
      "\nEvaluation RDS: ", directories$evaluation,
      "\nCSV tables: ", directories$tables,
      "\nStan plots: ", directories$diagnostics, "\n", sep = "")

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
