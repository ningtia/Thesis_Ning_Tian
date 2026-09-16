# =====================================================================
# make.R -- single replication entry point.
#
#   Rscript make.R              run every stage that has no up-to-date output
#   Rscript make.R --force      rerun everything from raw data
#   Rscript make.R --stage data run one stage only
#   Rscript make.R --dry-run    print the plan without executing it
#
# Stages (in dependency order)
#   sim-data      simulation_study_nnsv.R -> the three simulated datasets
#   sim-recovery  those datasets -> parameter and shape recovery tables (Ch. 6)
#   sim-prior     those datasets -> prior-vs-posterior tables and figures
#   sim-loo       those datasets -> the LOO detection calibration (Ch. 6)
#   data          raw data/*  ->  train/valid/test_dataset.csv, har_rv_*_dataset.csv
#   models        the split CSVs -> results/models/*.rds, results/tables/*.csv
#   analysis      the fitted models -> decomposition and regime-conditional tables
#   figures       the analysis series -> the two economic figures
#
# The four sim-* stages are OPT-IN: each is tens to hundreds of MCMC fits and
# runs for hours. A plain `make.R` reports whether their outputs are stale but
# will not rerun them. Add --with-simulation to actually run them, or name one
# explicitly with --stage, which counts as consent. Note that they source
# ../../nonlinear_sv.R, so editing that file marks them stale even when the
# Stan model itself (nonlinear_sv.stan, identical to the simulation copy
# nonlinear_sv_v3.stan) has not changed.
#
# NOTE: economic_results_pipeline.R is an earlier standalone script that writes
# to the same two figure filenames in results/economic/. It is superseded by
# stages `analysis` and `figures` and is deliberately NOT called here; running
# it afterwards would overwrite the thesis figures with the older styling.
#
# Determinism: every random draw in this pipeline is seeded. run_models.R sets
# GLOBAL_SEED and derives each model's seed from it; the Model Confidence Set
# bootstrap and the posterior predictive replications carry their own fixed
# seeds. 01_build_datasets.py draws no random numbers at all. Re-running from
# scratch on the same inputs reproduces every table byte for byte.
# =====================================================================

args <- commandArgs(trailingOnly = TRUE)
FORCE    <- "--force"   %in% args
DRY_RUN  <- "--dry-run" %in% args
WITH_SIM <- "--with-simulation" %in% args
STAGE    <- if ("--stage" %in% args) args[which(args == "--stage") + 1L] else "all"

PROJECT <- normalizePath(".", winslash = "/", mustWork = TRUE)
RLIB    <- file.path(PROJECT, "Rlib")
if (dir.exists(RLIB)) .libPaths(c(RLIB, .libPaths()))

PYTHON <- Sys.getenv("PYTHON", unset = "python")

say <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# A stage runs when forced, or when any of its outputs is missing or older than
# any of its inputs. Keeps a rerun cheap without hiding a stale result.
needs_run <- function(inputs, outputs) {
  if (FORCE) return(TRUE)
  if (!all(file.exists(outputs))) return(TRUE)
  present_inputs <- inputs[file.exists(inputs)]
  if (!length(present_inputs)) return(FALSE)
  max(file.mtime(present_inputs)) > min(file.mtime(outputs))
}

run_stage <- function(name, inputs, outputs, action, opt_in = FALSE) {
  if (!(STAGE %in% c("all", name))) return(invisible(FALSE))
  if (!needs_run(inputs, outputs)) {
    say("SKIP  ", name, " (outputs up to date)")
    return(invisible(FALSE))
  }
  # An expensive stage is only run on an explicit request. Naming it with
  # --stage counts as one; a bare `make.R` reports the staleness instead of
  # silently spending hours on it.
  if (opt_in && !WITH_SIM && !identical(STAGE, name)) {
    say("STALE ", name, " (opt-in; rerun with --with-simulation or --stage ", name, ")")
    return(invisible(FALSE))
  }
  if (DRY_RUN) {
    say("WOULD RUN  ", name)
    return(invisible(TRUE))
  }
  say("RUN   ", name)
  action()
  missing <- outputs[!file.exists(outputs)]
  if (length(missing)) {
    stop(name, " finished but did not produce: ",
         paste(basename(missing), collapse = ", "), call. = FALSE)
  }
  say("DONE  ", name)
  invisible(TRUE)
}

shell_run <- function(cmd, cmd_args) {
  status <- system2(cmd, cmd_args)
  if (!identical(status, 0L)) {
    stop(cmd, " ", paste(cmd_args, collapse = " "),
         " failed with status ", status, call. = FALSE)
  }
}

SPLIT_FILES <- file.path(PROJECT, c("train_dataset.csv", "valid_dataset.csv",
                                    "test_dataset.csv"))
HAR_FILES <- file.path(PROJECT, c("har_rv_daily_dataset.csv",
                                  "har_rv_weekly_dataset.csv"))

SIM <- file.path(PROJECT, "results", "simulation")
sim_f <- function(...) file.path(SIM, c(...))

# Sourced by recovery_replications.R and prior_predictive_check.R through
# ../../nonlinear_sv.R, which in turn sources benchmark_utils.R.
SIM_SHARED <- c(sim_f("nonlinear_sv_v3.stan"),
                file.path(PROJECT, c("nonlinear_sv.R", "benchmark_utils.R")))
SIM_DATA <- sim_f("nnsv_simulation_null.csv", "nnsv_simulation_main.csv",
                  "nnsv_simulation_strong.csv")

# Every simulation script uses paths relative to results/simulation, so each
# is sourced with chdir = TRUE.
run_sim <- function(script) source(file.path(SIM, script), chdir = TRUE)

# ---------------------------------------------------------------------
# Stage 0a: generate the three simulated datasets (zero / main / strong).
# Cheap on its own, but everything below depends on it.
# ---------------------------------------------------------------------
run_stage(
  "sim-data",
  inputs  = sim_f("simulation_study_nnsv.R"),
  # the script also writes nnsv_main_true_parameters.csv, which no chapter
  # uses and which is absent from the current results -- not required here
  outputs = SIM_DATA,
  action  = function() run_sim("simulation_study_nnsv.R"),
  opt_in  = TRUE
)

# ---------------------------------------------------------------------
# Stage 0b: replicated parameter and shape recovery -- Tables 6.2 to 6.5.
# Ten replications per scenario, four chains each.
# ---------------------------------------------------------------------
run_stage(
  "sim-recovery",
  inputs  = c(SIM_DATA, SIM_SHARED, sim_f("recovery_replications.R")),
  outputs = sim_f("nnsv_recovery_summary_s1_T520.csv",
                  "nnsv_recovery_shape_summary_s1_T520.csv",
                  "nnsv_recovery_attempt_summary_s1_T520.csv"),
  action  = function() run_sim("recovery_replications.R"),
  opt_in  = TRUE
)

# ---------------------------------------------------------------------
# Stage 0c: prior versus posterior for g_sd -- Figure 6.1. Each scenario is
# fitted twice, once with the likelihood switched off.
# ---------------------------------------------------------------------
run_stage(
  "sim-prior",
  inputs  = c(SIM_DATA, SIM_SHARED, sim_f("prior_predictive_check.R")),
  outputs = sim_f("nnsv_prior_vs_posterior_zero.png",
                  "nnsv_prior_vs_posterior_strong.png",
                  "nnsv_prior_vs_posterior_all.csv",
                  "nnsv_g_shape_all.csv"),
  action  = function() run_sim("prior_predictive_check.R"),
  opt_in  = TRUE
)

# ---------------------------------------------------------------------
# Stage 0d: can leave-one-out detect the nonlinearity? -- Table 6.6. Fits the
# network and the linear model to each simulated dataset and compares them
# exactly as Chapter 7 does on the observed data.
# ---------------------------------------------------------------------
run_stage(
  "sim-loo",
  inputs  = c(SIM_DATA, sim_f("null_test_loo.R", "nonlinear_sv_v3.stan",
                              "linear_sv.stan")),
  outputs = sim_f("nnsv_loo_null_test.csv"),
  action  = function() run_sim("null_test_loo.R"),
  opt_in  = TRUE
)

# ---------------------------------------------------------------------
# Stage 1: build the analysis datasets from the raw sources.
# ---------------------------------------------------------------------
run_stage(
  "data",
  inputs  = c(list.files(file.path(PROJECT, "raw data"), full.names = TRUE),
              file.path(PROJECT, "01_build_datasets.py"),
              file.path(PROJECT, "build_har_rv_daily_dataset.py")),
  outputs = c(SPLIT_FILES, HAR_FILES),
  action  = function() {
    shell_run(PYTHON, c(shQuote(file.path(PROJECT, "01_build_datasets.py")),
                        "--raw-dir", shQuote(file.path(PROJECT, "raw data")),
                        "--out-dir", shQuote(PROJECT),
                        "--summary"))
    # The HAR benchmark needs a weekly realised-variance proxy aggregated from
    # daily closes, which the weekly builder above does not produce.
    shell_run(PYTHON, c(shQuote(file.path(PROJECT, "build_har_rv_daily_dataset.py"))))
  }
)

# ---------------------------------------------------------------------
# Stage 2: estimate every model and write the evaluation tables.
# ---------------------------------------------------------------------
run_stage(
  "models",
  inputs  = c(SPLIT_FILES, HAR_FILES,
              file.path(PROJECT, c("run_models.R", "benchmark_utils.R",
                                   "evaluate_results.R", "GARCH.R", "HAR_RV.R",
                                   "linear_sv.R", "regime_switching_sv.R",
                                   "nonlinear_sv.R", "linear_sv.stan",
                                   "regime_switching_sv.stan",
                                   "nonlinear_sv.stan"))),
  outputs = file.path(PROJECT, "results", "tables",
                      c("overall_summary.csv", "architecture_selection.csv",
                        "stan_convergence.csv", "density_scores.csv",
                        "tail_backtests.csv", "portfolio_results.csv",
                        "posterior_predictive_checks.csv",
                        "loo_comparison.csv")),
  action  = function() source(file.path(PROJECT, "run_models.R"), chdir = TRUE)
)

# ---------------------------------------------------------------------
# Stage 3: post-estimation analysis (Chapter 8).
#
#   02  decomposition of the fitted nonlinearity -- ablation, partial
#       dependence, impulse responses
#   03  regime-conditional Diebold-Mariano tests, the portfolio table with
#       buy-and-hold, and the series behind the two economic figures
#
# Both read stored posterior draws and forecast tables; neither re-estimates
# anything, so the stage runs in minutes rather than hours.
# ---------------------------------------------------------------------
MODEL_RDS <- file.path(PROJECT, "results", "models",
                       paste0(c("sGARCH", "gjrGARCH", "eGARCH", "harRV",
                                "linearSV", "regimeSwitchingSV", "nonlinearSV"),
                              ".rds"))

run_stage(
  "analysis",
  inputs  = c(MODEL_RDS,
              file.path(PROJECT, c("02_decompose_nonlinearity.R",
                                   "03_conditional_economics.R",
                                   "benchmark_utils.R", "evaluate_results.R"))),
  outputs = c(file.path(PROJECT, "results", "tables",
                        c("nnsv_input_ablation.csv",
                          "nnsv_joint_ablation.csv",
                          "nnsv_partial_dependence.csv",
                          "nnsv_impulse_responses.csv",
                          "dm_by_cpu_regime.csv",
                          "portfolio_with_buyhold.csv")),
              file.path(PROJECT, "results", "economic",
                        c("wealth_paths.csv", "rolling_strategy_volatility.csv"))),
  action  = function() {
    source(file.path(PROJECT, "02_decompose_nonlinearity.R"), chdir = TRUE)
    source(file.path(PROJECT, "03_conditional_economics.R"), chdir = TRUE)
  }
)

# ---------------------------------------------------------------------
# Stage 4: the two economic figures. Kept separate from `analysis` so that
# the plots can be restyled without redoing the analysis behind them.
# ---------------------------------------------------------------------
run_stage(
  "figures",
  inputs  = c(file.path(PROJECT, "results", "economic",
                        c("wealth_paths.csv", "rolling_strategy_volatility.csv")),
              file.path(PROJECT, "04_economic_figures.R")),
  outputs = file.path(PROJECT, "results", "economic",
                      c("wealth_paths.pdf", "rolling_volatility.pdf")),
  action  = function() source(file.path(PROJECT, "04_economic_figures.R"),
                              chdir = TRUE)
)

say("make.R complete")
