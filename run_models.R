################################################################################
# run_models.R
# Run one model, several models, all models, or evaluation only.
################################################################################

# ========================== 1. EDIT ONLY HERE ================================

# "all"; one model; several models; or character(0) for evaluation only.
MODELS_TO_RUN <- "all"
# Examples:
# MODELS_TO_RUN <- "sGARCH"
# MODELS_TO_RUN <- c("sGARCH", "gjrGARCH", "eGARCH")
# MODELS_TO_RUN <- "linearSV"
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

# ========================== 2. FILE NAMES ====================================

first_file <- function(x, required = TRUE) {
  hit <- x[file.exists(x)]
  if (length(hit)) return(hit[1L])
  if (required) stop("File not found: ", paste(x, collapse = " / "), call. = FALSE)
  NULL
}

FILES <- list(
  utils = first_file("benchmark_utils.R"),
  garch = first_file(c("GARCH.R"), FALSE),
  linear = first_file(c("linear_sv.R"), FALSE),
  ms = first_file(c("regime_switching_sv.R"), FALSE),
  har = first_file(c("HAR_RV.R"), FALSE),
  nn = first_file(c("nonlinear_sv.R"), FALSE),

  evaluation = first_file(c("evaluate_results.R"), FALSE)
)

# ========================== 3. GLOBAL OPTIONS ================================

# These stop model files from automatically running when sourced.
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
  benchmark.stan_sampling = STAN_SAMPLING,
  benchmark.stan_adapt_delta = STAN_ADAPT_DELTA,
  benchmark.stan_max_treedepth = STAN_MAX_TREEDEPTH,

  benchmark.linear_chains = STAN_CHAINS,
  benchmark.linear_warmup = STAN_WARMUP,
  benchmark.linear_sampling = STAN_SAMPLING,
  benchmark.linear_adapt_delta = STAN_ADAPT_DELTA,
  benchmark.linear_max_treedepth = STAN_MAX_TREEDEPTH,

  benchmark.ms_chains = STAN_CHAINS,
  benchmark.ms_warmup = STAN_WARMUP,
  benchmark.ms_sampling = STAN_SAMPLING,
  benchmark.ms_adapt_delta = STAN_ADAPT_DELTA,
  benchmark.ms_max_treedepth = STAN_MAX_TREEDEPTH,

  benchmark.nn_chains = STAN_CHAINS,
  benchmark.nn_warmup = STAN_WARMUP,
  benchmark.nn_sampling = STAN_SAMPLING,
  benchmark.nn_adapt_delta = STAN_ADAPT_DELTA,
  benchmark.nn_max_treedepth = STAN_MAX_TREEDEPTH,
  benchmark.nn_compute_loo = NN_COMPUTE_LOO,
  benchmark.architecture_metric = NN_ARCHITECTURE_METRIC,
  benchmark.nn_seed = GLOBAL_SEED
)

if (!requireNamespace("rstan", quietly = TRUE)) {
  stop("Install rstan first.", call. = FALSE)
}
if (!requireNamespace("posterior", quietly = TRUE)) {
  stop("Install posterior first.", call. = FALSE)
}
if (!requireNamespace("bayesplot", quietly = TRUE)) {
  stop("Install bayesplot first.", call. = FALSE)
}

rstan::rstan_options(auto_write = TRUE)
cores <- parallel::detectCores(logical = TRUE)
if (is.na(cores)) cores <- 1L
options(mc.cores = min(STAN_CHAINS, cores))
set.seed(GLOBAL_SEED)

# ========================== 4. LOAD FUNCTIONS AND DATA =======================

source(FILES$utils)
for (f in c(FILES$garch, FILES$linear, FILES$ms, FILES$nn, FILES$har)) {
  if (!is.null(f)) {
    message("Sourcing ", f)
    source(f)
  }
}

bench <- load_benchmark_data()

if (!all(c("all", "y_all", "X_all", "split_rows") %in% names(bench))) {
  stop("bench is missing all/y_all/X_all/split_rows.", call. = FALSE)
}

# ========================== 5. OUTPUT DIRECTORIES ============================

DIR <- list(
  models = "results/models",
  evaluation = "results/evaluation",
  tables = "results/tables",
  diagnostics = "results/diagnostics"
)
invisible(lapply(DIR, dir.create, recursive = TRUE, showWarnings = FALSE))

# ========================== 6. FLEXIBLE FUNCTION CALLING =====================

first_fun <- function(candidates, required = TRUE) {
  ok <- candidates[vapply(candidates, exists, logical(1),
                          mode = "function", inherits = TRUE)]
  if (length(ok)) return(ok[1L])
  if (required) {
    stop("Runner not found: ", paste(candidates, collapse = " / "),
         call. = FALSE)
  }
  NULL
}

# Pass only arguments that the actual runner accepts.
call_flexible <- function(candidates, args, aliases = list(),
                          required = TRUE) {
  nm <- first_fun(candidates, required)
  if (is.null(nm)) return(NULL)

  fn <- get(nm, mode = "function", inherits = TRUE)
  fml <- names(formals(fn))
  dots <- "..." %in% fml
  use <- list()

  for (key in names(args)) {
    possible <- unique(c(key, aliases[[key]]))
    target <- possible[possible %in% fml]
    if (length(target)) {
      use[[target[1L]]] <- args[[key]]
    } else if (dots) {
      use[[key]] <- args[[key]]
    }
  }

  message("Calling ", nm, "(", paste(names(use), collapse = ", "), ")")
  do.call(fn, use)
}

extract_result <- function(x, key, aliases = character()) {
  if (is.list(x) && !is.null(x$forecast_table)) return(x)
  for (nm in unique(c(key, aliases))) {
    if (is.list(x[[nm]]) && !is.null(x[[nm]]$forecast_table)) return(x[[nm]])
    if (is.list(x$benchmark_results[[nm]]) &&
        !is.null(x$benchmark_results[[nm]]$forecast_table)) {
      return(x$benchmark_results[[nm]])
    }
  }
  stop("No forecast_table found for ", key, ".", call. = FALSE)
}

check_result <- function(x, key) {
  if (!is.list(x) || !is.data.frame(x$forecast_table) ||
      nrow(x$forecast_table) == 0L) {
    stop(key, " did not return a non-empty forecast_table.", call. = FALSE)
  }
  needed <- c("forecast_date", "split", "actual_return", "variance",
              "VaR_01", "ES_01", "VaR_05", "ES_05", "log_score")
  missing <- setdiff(needed, names(x$forecast_table))
  if (length(missing)) {
    stop(key, " forecast_table missing: ",
         paste(missing, collapse = ", "), call. = FALSE)
  }
  invisible(TRUE)
}

COMMON_ALIASES <- list(
  bench = c("data", "benchmark_data"),
  model = c("garch_model", "variance_model", "model_name", "spec_model"),
  refit_every = c("refit", "refit_frequency"),
  save_fits = c("save_rolling_fits"),
  seed = c("random_seed")
)

# ========================== 7. MODEL ADAPTERS ================================

run_garch_one <- function(key) {
  single <- first_fun(c("run_garch_benchmark", "run_garch_model",
                        "run_single_garch", "run_garch"), FALSE)

  if (!is.null(single)) {
    out <- call_flexible(
      single,
      list(bench = bench, model = key, refit_every = REFIT_EVERY,
           save_fits = SAVE_ALL_ROLLING_FITS, seed = GLOBAL_SEED),
      COMMON_ALIASES
    )
  } else {
    out <- call_flexible(
      c("run_garch_benchmarks", "run_all_garch_models", "run_all_garch"),
      list(bench = bench, refit_every = REFIT_EVERY,
           save_fits = SAVE_ALL_ROLLING_FITS, seed = GLOBAL_SEED),
      COMMON_ALIASES
    )
  }

  aliases <- switch(
    key,
    sGARCH = c("GARCH", "garch"),
    gjrGARCH = c("GJR-GARCH", "gjr"),
    eGARCH = c("EGARCH", "egarch"),
    character()
  )
  extract_result(out, key, aliases)
}

run_linear <- function() {
  out <- call_flexible(
    c("run_linear_sv", "run_linear_sv_benchmark", "run_linearSV"),
    list(bench = bench, refit_every = REFIT_EVERY,
         save_fits = SAVE_ALL_ROLLING_FITS, seed = GLOBAL_SEED),
    COMMON_ALIASES
  )
  extract_result(out, "linearSV", c("linear_sv", "LinearSV"))
}

run_ms <- function() {
  out <- call_flexible(
    c("run_regime_switching_sv", "run_regime_switching_sv_benchmark",
      "run_ms_sv", "run_ms_sv_benchmark", "run_MSSV"),
    list(bench = bench, refit_every = REFIT_EVERY,
         save_fits = SAVE_ALL_ROLLING_FITS, seed = GLOBAL_SEED),
    COMMON_ALIASES
  )
  extract_result(out, "regimeSwitchingSV",
                 c("msSV", "MSSV", "regime_switching_sv"))
}

run_nn <- function() {
  if (!exists("run_nonlinear_sv", mode = "function")) {
    stop("run_nonlinear_sv() not found.", call. = FALSE)
  }

  args <- list(bench = bench,
               architecture_metric = NN_ARCHITECTURE_METRIC)

  if (exists("default_nn_candidates", mode = "function")) {
    args$candidates <- default_nn_candidates()
  }

  if (exists("nn_sv_default_config", mode = "function")) {
    cfg <- nn_sv_default_config()
    cfg$refit_every <- REFIT_EVERY
    cfg$save_fits <- SAVE_ALL_ROLLING_FITS
    cfg$seed <- GLOBAL_SEED
    cfg$chains <- STAN_CHAINS
    cfg$warmup <- STAN_WARMUP
    if ("sampling" %in% names(cfg)) cfg$sampling <- STAN_SAMPLING
    if ("iter" %in% names(cfg)) cfg$iter <- STAN_WARMUP + STAN_SAMPLING
    cfg$adapt_delta <- STAN_ADAPT_DELTA
    cfg$max_treedepth <- STAN_MAX_TREEDEPTH
    cfg$compute_loo <- NN_COMPUTE_LOO
    args$config <- cfg
  }

  extract_result(do.call(run_nonlinear_sv, args), "nonlinearSV",
                 c("nnSV", "NNSV"))
}

run_har <- function() {
  out <- call_flexible(
    c("run_har_rv", "run_har_rv_benchmark", "run_HAR_RV"),
    list(bench = bench, refit_every = REFIT_EVERY,
         save_fits = SAVE_ALL_ROLLING_FITS, seed = GLOBAL_SEED),
    COMMON_ALIASES,
    required = FALSE
  )
  if (is.null(out)) return(NULL)
  extract_result(out, "harRV", c("HAR", "HAR_RV", "har_rv"))
}

REGISTRY <- list(
  sGARCH = list(file = FILES$garch, run = function() run_garch_one("sGARCH")),
  gjrGARCH = list(file = FILES$garch,
                  run = function() run_garch_one("gjrGARCH")),
  eGARCH = list(file = FILES$garch, run = function() run_garch_one("eGARCH")),
  linearSV = list(file = FILES$linear, run = run_linear),
  regimeSwitchingSV = list(file = FILES$ms, run = run_ms),
  nonlinearSV = list(file = FILES$nn, run = run_nn),
  harRV = list(file = FILES$har, run = run_har, optional = TRUE)
)

ALL_KEYS <- names(REGISTRY)

resolve_selection <- function(x) {
  if (!length(x)) return(character())
  if ("all" %in% x) {
    return(ALL_KEYS[vapply(REGISTRY, function(z) !is.null(z$file), logical(1))])
  }
  bad <- setdiff(x, ALL_KEYS)
  if (length(bad)) stop("Unknown model(s): ", paste(bad, collapse = ", "))
  unique(x)
}
SELECTED <- resolve_selection(MODELS_TO_RUN)

# ========================== 8. LOAD SAVED RESULTS ============================

model_path <- function(key) file.path(DIR$models, paste0(key, ".rds"))

load_saved <- function() {
  ans <- list()
  for (key in ALL_KEYS) {
    p <- model_path(key)
    if (file.exists(p)) {
      x <- tryCatch(readRDS(p), error = function(e) NULL)
      if (!is.null(x)) {
        ok <- tryCatch({check_result(x, key); TRUE}, error = function(e) FALSE)
        if (ok) ans[[key]] <- x
      }
    }
  }
  ans
}

benchmark_results <- if (LOAD_SAVED_RESULTS) load_saved() else list()

# ========================== 9. RUN SELECTED MODELS ===========================

errors <- list()

for (key in SELECTED) {
  p <- model_path(key)

  if (is.null(REGISTRY[[key]]$file)) {
    message("Skipping ", key, ": source file not found.")
    next
  }

  if (!FORCE_RERUN && file.exists(p)) {
    message("Skipping ", key, ": saved result exists. ",
            "Set FORCE_RERUN <- TRUE to rerun.")
    benchmark_results[[key]] <- readRDS(p)
    next
  }

  message("\n", strrep("=", 70), "\nRUNNING: ", key, "\n", strrep("=", 70))
  started <- Sys.time()

  x <- tryCatch(
    REGISTRY[[key]]$run(),
    error = function(e) {
      errors[[key]] <<- conditionMessage(e)
      warning(key, " failed: ", conditionMessage(e), call. = FALSE)
      NULL
    }
  )

  if (!is.null(x)) {
    check_result(x, key)
    benchmark_results[[key]] <- x
    saveRDS(x, p, compress = TRUE)
    mins <- as.numeric(difftime(Sys.time(), started, units = "mins"))
    message("Saved ", key, " to ", p, " (", round(mins, 2), " min).")
  }
}

# Reload all valid saved results after the run.
if (LOAD_SAVED_RESULTS) {
  disk <- load_saved()
  benchmark_results[names(disk)] <- disk
}

saveRDS(benchmark_results,
        file.path(DIR$models, "benchmark_results_all_available.rds"),
        compress = TRUE)

# ========================== 10. FINAL STAN PLOTS =============================

save_plot <- function(key, pars, pairs) {
  x <- benchmark_results[[key]]
  if (is.null(x$fit) || !inherits(x$fit, "stanfit")) return(invisible(FALSE))
  if (!exists("save_stan_diagnostic_plots", mode = "function")) {
    return(invisible(FALSE))
  }
  save_stan_diagnostic_plots(
    fit = x$fit,
    file = file.path(DIR$diagnostics, paste0(key, "_final_refit.pdf")),
    pars = pars,
    pairs_pars = pairs
  )
  invisible(TRUE)
}

if (SAVE_FINAL_STAN_PLOTS) {
  save_plot("linearSV",
            c("mu", "phi", "phi_raw", "sigma_eta", "nu_minus2"),
            c("phi", "phi_raw", "sigma_eta", "nu_minus2"))
  save_plot("regimeSwitchingSV",
            c("mu", "phi", "sigma_eta", "p11", "p22", "nu_minus2"),
            c("phi", "sigma_eta", "p11", "p22"))
  save_plot("nonlinearSV",
            c("mu", "phi", "phi_raw", "sigma_eta", "tau_w", "s"),
            c("phi", "phi_raw", "sigma_eta", "tau_w", "s"))
}

# ========================== 11. EVALUATE ALL AVAILABLE RESULTS ===============

write_df <- function(x, name) {
  if (is.data.frame(x)) {
    utils::write.csv(x, file.path(DIR$tables, name), row.names = FALSE)
  }
}

evaluation_results <- NULL

if (RUN_EVALUATION && length(benchmark_results)) {
  if (is.null(FILES$evaluation)) {
    warning("Evaluation file not found.", call. = FALSE)
  } else {
    source(FILES$evaluation)

    evaluation_results <- evaluate_all_models(
      bench = bench,
      benchmark_results = benchmark_results,
      cpu_variable = CPU_VARIABLE,
      mcs_bootstrap = MCS_BOOTSTRAP,
      ppc_replications = PPC_REPLICATIONS
    )

    saveRDS(evaluation_results,
            file.path(DIR$evaluation,
                      "evaluation_results_all_available.rds"),
            compress = TRUE)

    write_df(evaluation_results$summary, "overall_summary.csv")
    write_df(evaluation_results$architecture_selection,
             "architecture_selection.csv")
    write_df(evaluation_results$convergence, "stan_convergence.csv")
    write_df(evaluation_results$density_scores, "density_scores.csv")
    write_df(evaluation_results$tail_backtests, "tail_backtests.csv")
    write_df(evaluation_results$portfolio, "portfolio_results.csv")
    write_df(evaluation_results$posterior_predictive_checks,
             "posterior_predictive_checks.csv")

    if (is.list(evaluation_results$dm_tests)) {
      write_df(evaluation_results$dm_tests$QLIKE, "dm_qlike.csv")
      write_df(evaluation_results$dm_tests$negative_LPS,
               "dm_negative_lps.csv")
      write_df(evaluation_results$dm_tests$FZ0_5pct, "dm_fz05.csv")
    }
  }
}

# ========================== 12. CONSOLE SUMMARY ==============================

cat("\n", strrep("=", 70), "\nRUN COMPLETE\n", strrep("=", 70), "\n", sep = "")
cat("Available results: ",
    if (length(benchmark_results)) paste(names(benchmark_results),
                                         collapse = ", ") else "<none>",
    "\n", sep = "")

if (length(errors)) {
  cat("Failures:\n")
  for (key in names(errors)) {
    cat("  - ", key, ": ", errors[[key]], "\n", sep = "")
  }
}

if (!is.null(evaluation_results)) {
  cat("\nOverall evaluation:\n")
  print(evaluation_results$summary)
}

cat("\nModel RDS: ", DIR$models,
    "\nEvaluation RDS: ", DIR$evaluation,
    "\nCSV tables: ", DIR$tables,
    "\nStan plots: ", DIR$diagnostics, "\n", sep = "")
