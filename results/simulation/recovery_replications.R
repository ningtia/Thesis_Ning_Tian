# recovery_replications.R
# A single simulated dataset cannot establish parameter recovery: "the truth is
# inside the 95% interval" is a coin flip you win 95% of the time by luck.
# This repeats simulate-and-fit over R datasets and reports bias, RMSE and
# empirical coverage, which is what a referee will ask for.
#
# The DGP below mirrors simulation_study_nnsv.R's "recommended design"
# (x_ar = 0.90, burn_in = 200): a persistent covariate is what makes g_t
# identifiable from i.i.d. state noise in the first place, and it is the
# design nnsv_simulation_main.csv -- and therefore Simulation_recovery.R's
# single-fit diagnostic -- actually uses.
#
# Because a persistent x makes mean(g) != 0 in any finite sample, mu's "true"
# value is not the DGP intercept 1.80 -- it is level-shifted by
# mean(g)/(1-phi), same correction as nnsv_main_true_parameters.csv. That
# shift (and true g_sd = sd(g)) is realization-specific, so both are
# recomputed from each replication's own simulated path rather than fixed
# once, matching how Simulation_recovery.R reads its true values off the one
# dataset it fits.
#
# h_bar (sample mean of the log-variance path) is tracked alongside mu: with
# phi_true = 0.95 close to its ceiling, mu individually sits on a ridge with
# phi and can recover badly even in a fit that is otherwise fine, while
# h_bar -- the level the application actually cares about -- stays
# identified. See nonlinear_sv_v3.stan's header comment on h_bar.
#
# 2026-08-22, three methodology gaps closed after review:
#
#   1. g_sd's coverage_95 is not a meaningful pass/fail check for the zero
#      scenario. g_sd = sd(g) >= 0 always, and the prior over it is
#      continuous, so a well-behaved posterior essentially never puts the
#      exact boundary value 0 inside its 95% interval -- coverage_95 there
#      trends to ~0 regardless of how good the model is. summary_tbl now
#      also reports mean_prob_below_delta (posterior P(g_sd < G_SD_DELTA),
#      averaged over replications) and frac_upper_ci_below_delta (how often
#      the 95% upper bound itself already rules out a bigger nonlinearity)
#      for the g_sd row specifically -- read those, not coverage_95, for the
#      false-positive check.
#
#   2. The g PATH shape (correlation/RMSE of the fitted g against the true
#      centred path) used to be checked only once, in
#      prior_predictive_check.R -- so only g_sd's SIZE was ever shown to
#      recover reliably, never the shape. Every replication now also fits
#      draws$g and records shape correlation/RMSE (shape_reps / the
#      "_shape" files); zero's true g path is exactly constant, so its
#      correlation is undefined and reported as NA, matching
#      prior_predictive_check.R's convention, with RMSE alone standing in as
#      the spurious-signal measure.
#
#   3. Aggregating only usable == TRUE fits before computing bias/RMSE/
#      coverage is correct (an unconverged posterior is not a posterior
#      summary), but silently doing that without also reporting how many
#      replications were thrown away invites "recovery looks good because
#      the hard datasets got dropped." Every scenario/T combination now also
#      writes an attempt-level table: n_attempted, n_usable, usable_rate,
#      mean/median divergences (over ALL attempts, not just usable ones),
#      and separate rhat/ESS-bulk/ESS-tail/E-BFMI failure counts, via
#      benchmark_utils.R's stan_fit_diagnostics() (already available through
#      the nonlinear_sv.R source below) instead of the narrower divergences+
#      rhat+ess_bulk-only check this used to run. summary_tbl itself now
#      loops over the fixed `targets` list rather than split()'s dynamically
#      discovered groups, so a combination where n_usable = 0 still gets a
#      visible row (NAs, not a silently missing scenario/T) instead of
#      vanishing from the combined CSV.
#
# Two things get replicated here, in sequence:
#
#   1. Three DGP scenarios (zero/main/strong -- same g_scale/lev_scale as
#      prior_predictive_check.R) at one T, so identifiability claims from a
#      single prior_predictive_check.R fit per scenario don't rest on a coin
#      flip. All three now write "_<scenario>" files (main included -- see
#      point 3 above, the output schema changed enough that keeping main's
#      old bare filenames as a special case wasn't worth the extra code); a
#      combined "_all" summary reads all three side by side.
#
#   2. 2026-08-22, per advisor request: "repeat the simulation with three
#      sizes: T = 312, T = 520 (our full sample), and T = 1000 (just to see
#      what more data would give us). Then we can explain WHY we chose T,
#      and it will not look like an accident." All three scenarios (zero/
#      main/strong) x all three T's, with K = 4 and the same priors held
#      fixed across T, so the comparison isolates the effect of T alone.
#      Output goes to "nnsv_recovery_summary_by_T.csv": read bias shrinking
#      and coverage_95 moving toward 0.95 as T grows 312 -> 520 -> 1000 as
#      the "more data helps, and 520 already captures most of the benefit"
#      argument -- for main that settles "why T = 520"; for zero/strong it
#      also shows whether false-positive control and the identifiability
#      stress test hold up at T = 520 or only kick in at T = 1000. If 1000
#      keeps improving a lot over 520, report that honestly too -- a real
#      limitation of the sample, not something to hide.
#
# Run time is R x (3 scenarios + 3 T's) x (one fit) -- T = 1000 fits are the
# slow ones. R_REP below applies to both parts; start with 5-20, 50-100 is
# better once you know nothing is on fire.

library(rstan)
library(posterior)

rstan_options(auto_write = TRUE)

# Warm-start init (data-driven mu/phi/sigma_eta, NN weights near zero) and
# stan_fit_diagnostics() (full rhat/ESS-bulk/ESS-tail/E-BFMI/divergence
# bundle), both reused from nonlinear_sv.R / benchmark_utils.R -- same fix
# that took the single-fit scenarios in prior_predictive_check.R from
# double-digit divergences down to essentially none. See
# prior_predictive_check.R for why the autorun option and chdir matter here.
options(benchmark.nn_autorun = FALSE)
source("../../nonlinear_sv.R", chdir = TRUE)

options(mc.cores = 4)
rstan_options(auto_write = TRUE)
rstan_options(threads_per_chain = 1) # 必须保持为1
# Toggle which part(s) actually run without deleting either -- e.g. set
# RUN_SCENARIOS <- FALSE to rerun just the T-sensitivity part once Part 1's
# results are already on disk, instead of redoing fits you already have.
RUN_SCENARIOS      <- FALSE
RUN_T_SENSITIVITY  <- FALSE
RUN_S1_FIX         <- TRUE

R_REP     <- 10
K_hidden  <- 4
X_AR      <- 0.90   # covariate persistence; matches simulation_study_nnsv.R's recommended design
BURN_IN   <- 200
G_SD_DELTA <- 0.05  # "practically zero" threshold for the g_sd false-positive check (point 1 above)

# 2026-08-23, advisor's usable-rule fix: judge convergence only on the
# quantities actually used downstream -- g_sd, h_bar, phi, sigma_eta, nu, and
# the log-likelihood (lp__) -- NOT raw NN weights (label-switching makes their
# individual rhat meaningless) and NOT mu (h_bar replaces it: mu sits on a
# ridge with phi as phi -> 1 and can mix badly even when the fit is fine).
# Checked against the already-saved T-sensitivity attempts CSVs: every
# usable == FALSE case there was divergence-driven, not rhat/ess-driven, so
# this narrower gate does not retroactively change those numbers -- but it's
# the rule new runs (like the s = 1.0 fix below) should use going forward.
#
# lp__ is NOT in NEW_GATE_PARS: stan_fit_diagnostics() filters its `pars`
# argument through intersect(pars, fit@model_pars), and lp__ is a sampler
# quantity, not a declared model parameter, so it is never in fit@model_pars
# and would silently be dropped if listed here. It is checked separately
# below (lp_rhat/lp_ess_bulk/lp_ess_tail) and folded into `ok` by hand.
NEW_GATE_PARS <- c("g_sd", "h_bar", "phi", "sigma_eta", "nu")

mu_true        <- 1.80
phi_true       <- 0.95
sigma_eta_true <- 0.25
nu_true        <- 10

simulate_nnsv <- function(seed, T_sim, g_scale, lev_scale = 0, x_ar = X_AR, burn_in = BURN_IN) {
  set.seed(seed)
  n <- burn_in + T_sim

  # Persistent covariate: AR(1) with unit stationary variance. This
  # persistence is what makes g_t distinguishable from i.i.d. state noise
  # (sigma_eta * eta_t) -- an i.i.d. x does not identify g at this T.
  x     <- numeric(n)
  x[1]  <- rnorm(1)
  innov <- rnorm(n) * sqrt(1 - x_ar^2)
  for (t in 2:n) x[t] <- x_ar * x[t - 1] + innov[t]

  eta     <- rnorm(n)
  epsilon <- rt(n, df = nu_true)   # raw Student-t scale, matches the Stan likelihood

  h <- numeric(n); y <- numeric(n); g <- numeric(n)
  h[1] <- mu_true + sigma_eta_true / sqrt(1 - phi_true^2) * eta[1]
  y[1] <- exp(h[1] / 2) * epsilon[1]
  for (t in 2:n) {
    g[t] <- g_scale * sin(x[t - 1]) + lev_scale * min(y[t - 1], 0)
    h[t] <- mu_true + phi_true * (h[t - 1] - mu_true) + g[t] +
      sigma_eta_true * eta[t]
    y[t] <- exp(h[t] / 2) * epsilon[t]
  }

  keep <- (burn_in + 1):n
  data.frame(time = 1:T_sim, y = y[keep], x = x[keep],
             h_true = h[keep], g_true = g[keep])
}

model <- stan_model("nonlinear_sv_v3.stan")

targets <- c("mu", "phi", "sigma_eta", "nu", "g_sd", "h_bar")

# Same g_scale/lev_scale as simulation_study_nnsv.R's (b) main / (c) null /
# (d, renamed) strong designs, so this replicates exactly what
# prior_predictive_check.R fits once per scenario. Distinct seed_offsets
# keep the three scenarios' simulated datasets from overlapping.
scenarios <- list(
  main   = list(g_scale = 0.25, lev_scale = 0.00, seed_offset = 20260820),
  zero   = list(g_scale = 0.00, lev_scale = 0.00, seed_offset = 20261820),
  strong = list(g_scale = 0.65, lev_scale = 0.15, seed_offset = 20262820)
)

# ---------------------------------------------------------------------------
# Run R replications of one DGP scenario (g_scale/lev_scale) at one T end to
# end: simulate, fit (warm-started), record scalar recovery + g-shape
# recovery + divergence-location + full attempt-level diagnostics, then
# aggregate each into its own summary.
# ---------------------------------------------------------------------------
run_scenario_replications <- function(label, T_sim, g_scale, lev_scale, seed_offset) {
  results       <- list()
  shape_results <- list()
  attempt_diag  <- list()
  div_loc_all   <- list()

  for (r in 1:R_REP) {
    d <- simulate_nnsv(seed = seed_offset + r, T_sim = T_sim, g_scale = g_scale, lev_scale = lev_scale)

    # Realized true values for this replication's path (see header comment):
    # phi, sigma_eta, nu are DGP constants, but mu and g_sd depend on the
    # realized g and must be recomputed per replication.
    g_r <- d$g_true[2:T_sim]
    true_g_c <- g_r - mean(g_r)   # centred, matches prior_predictive_check.R
    true_vec_r <- c(
      mu        = mu_true + mean(g_r) / (1 - phi_true),
      phi       = phi_true,
      sigma_eta = sigma_eta_true,
      nu        = nu_true,
      g_sd      = sd(g_r),
      h_bar     = mean(d$h_true)
    )

    stan_data <- list(
      T = T_sim, D = 1, K = K_hidden,
      y = d$y,
      X = matrix(d$x[1:(T_sim - 1)], ncol = 1),
      use_student_t = 1,
      x_forecast = array(d$x[T_sim], dim = 1),
      s_fixed = 1, tau_w_scale = 0.2,
      mu_scale = 5, phi_a = 20, phi_b = 2,
      sigma_eta_scale = 1, b1_scale = 0.5, nu_rate = 0.1,
      prior_only = 0L,        # 0 = 正常后验估计;1 = 仅先验预测检查
      stationary_init = 0L,   # 使用 v3 的固定尺度初始状态
      h1_scale = 2.0          # h_1 相对 mu 的固定扩散尺度
    )

    linear_start <- nn_sv_linear_start(stan_data$y, stan_data$X)
    init_fn <- function() nn_sv_init(stan_data, previous_post = NULL, linear_start = linear_start)

    fit <- sampling(
      model, data = stan_data, chains = 4, cores = 4,
      iter = 3000, warmup = 1500, seed = 1000 + r, refresh = 0, init = init_fn,
      control = list(adapt_delta = 0.99, max_treedepth = 12)
    )

    # -------------------------------------------------------------------------
    # Full attempt-level diagnostics (point 3): rhat/ESS-bulk/ESS-tail/E-BFMI/
    # divergences all from stan_fit_diagnostics(), narrowed to the scalar
    # structural parameters (not the T-length h/eta_raw/g paths) so this
    # stays fast at T = 1000. treedepth hits aren't part of that helper, so
    # they're added separately from the same sampler_params object.
    # -------------------------------------------------------------------------
    diagnostics <- stan_fit_diagnostics(fit, pars = NEW_GATE_PARS)
    sp   <- get_sampler_params(fit, inc_warmup = FALSE)
    max_td <- 12
    max_treedepth_hit <- sum(sapply(sp, function(p) sum(p[, "treedepth__"] >= max_td)))
    ndiv <- diagnostics$divergence_sum

    # lp__ (the log-likelihood the advisor asked to check) has to be pulled
    # and checked by hand -- see the NEW_GATE_PARS comment above for why
    # stan_fit_diagnostics() can't do it via its `pars` argument.
    lp_draws    <- rstan::extract(fit, pars = "lp__", permuted = FALSE, inc_warmup = FALSE)
    lp_vec      <- lp_draws[, , "lp__"]
    lp_rhat     <- as.numeric(posterior::rhat(lp_vec))
    lp_ess_bulk <- as.numeric(posterior::ess_bulk(lp_vec))
    lp_ess_tail <- as.numeric(posterior::ess_tail(lp_vec))
    lp_ok <- is.finite(lp_rhat) && lp_rhat < 1.01 &&
      is.finite(lp_ess_bulk) && lp_ess_bulk > 400 &&
      is.finite(lp_ess_tail) && lp_ess_tail > 400

    ok <- isTRUE(diagnostics$all_checks_passed) && lp_ok

    attempt_diag[[length(attempt_diag) + 1]] <- data.frame(
      scenario = label, T = T_sim, rep = r,
      divergences       = ndiv,
      max_treedepth_hit = max_treedepth_hit,
      max_rhat          = diagnostics$max_rhat,
      min_ess_bulk      = diagnostics$min_bulk_ess,
      min_ess_tail      = diagnostics$min_tail_ess,
      min_ebfmi         = diagnostics$min_ebfmi,
      lp_rhat           = lp_rhat,
      lp_ess_bulk       = lp_ess_bulk,
      lp_ess_tail       = lp_ess_tail,
      divergences_ok    = isTRUE(diagnostics$checks$divergences_ok),
      rhat_ok           = isTRUE(diagnostics$checks$rhat_ok),
      ess_bulk_ok       = isTRUE(diagnostics$checks$bulk_ess_ok),
      ess_tail_ok       = isTRUE(diagnostics$checks$tail_ess_ok),
      ebfmi_ok          = isTRUE(diagnostics$checks$ebfmi_ok),
      lp_ok             = lp_ok,
      usable            = ok,
      row.names = NULL
    )

    # -------------------------------------------------------------------------
    # Where do the divergent transitions sit?  For each candidate quantity,
    # compare its mean among divergent draws to its mean over all draws, in
    # units of that quantity's overall sd (a numeric stand-in for
    # pairs(fit, condition = "divergent__")). Large |z_shift| = divergences
    # concentrate there; this is what usually fingers a funnel (e.g. tau_w
    # collapsing toward 0 while W1/w2_raw blow up) or a boundary ridge
    # (phi -> 1, mu-phi).
    # -------------------------------------------------------------------------
    if (ndiv > 0) {
      # permuted = FALSE keeps [iteration, chain, parameter] order aligned with
      # get_sampler_params(..., inc_warmup = FALSE); rstan::extract()'s default
      # permuted = TRUE shuffles draws and would misalign them against div_flag.
      draws_arr <- rstan::extract(fit, permuted = FALSE, inc_warmup = FALSE)
      div_flag  <- unlist(lapply(sp, function(p) p[, "divergent__"]))

      monitor_pars <- c("mu", "phi", "sigma_eta", "tau_w", "g_sd", "nu", "h_bar", "lp__")
      get_vec <- function(nm) as.vector(draws_arr[, , nm])

      w1_draws <- rstan::extract(fit, pars = "W1", permuted = FALSE, inc_warmup = FALSE)
      w2_draws <- rstan::extract(fit, pars = "w2", permuted = FALSE, inc_warmup = FALSE)
      extra <- list(
        max_abs_W1 = as.vector(apply(w1_draws, c(1, 2), function(v) max(abs(v)))),
        sum_w2     = as.vector(apply(w2_draws, c(1, 2), sum))
      )

      all_vars <- c(stats::setNames(lapply(monitor_pars, get_vec), monitor_pars), extra)

      div_loc <- do.call(rbind, lapply(names(all_vars), function(nm) {
        v <- all_vars[[nm]]
        data.frame(
          scenario = label, T = T_sim, rep = r, parameter = nm,
          mean_divergent = mean(v[div_flag == 1]),
          mean_all       = mean(v),
          sd_all         = sd(v),
          z_shift        = (mean(v[div_flag == 1]) - mean(v)) / sd(v),
          row.names = NULL
        )
      }))
      div_loc <- div_loc[order(-abs(div_loc$z_shift)), ]
      div_loc_all[[length(div_loc_all) + 1]] <- div_loc
      message("  [", label, " T=", T_sim, "] divergences (", ndiv, ") concentrate on: ",
              paste(sprintf("%s (z=%.2f)", head(div_loc$parameter, 3), head(div_loc$z_shift, 3)),
                    collapse = ", "))
    }

    dr <- rstan::extract(fit, pars = targets)
    for (p in targets) {
      v <- as.numeric(dr[[p]])
      # prob_below_delta only means anything for g_sd (the false-positive
      # check, point 1); NA elsewhere but the column has to exist on every
      # row for rbind() to line up.
      prob_below_delta <- if (p == "g_sd") mean(v < G_SD_DELTA) else NA_real_
      results[[length(results) + 1]] <- data.frame(
        scenario = label, T = T_sim, rep = r, parameter = p, true_value = true_vec_r[[p]],
        median = median(v), sd = sd(v),
        lower_95 = quantile(v, 0.025), upper_95 = quantile(v, 0.975),
        prob_below_delta = prob_below_delta,
        divergences = ndiv, usable = ok, row.names = NULL
      )
    }

    # -------------------------------------------------------------------------
    # Shape of g (point 2): repeated every replication, not just once in
    # prior_predictive_check.R. sd(true_g_c) == 0 in the zero scenario (g is
    # exactly constant there), so correlation is undefined -- reported as NA,
    # RMSE alone stands in as the spurious-fitted-signal measure.
    # -------------------------------------------------------------------------
    g_draws  <- rstan::extract(fit, pars = "g")$g
    g_median <- apply(g_draws, 2, median)
    shape_results[[length(shape_results) + 1]] <- data.frame(
      scenario = label, T = T_sim, rep = r,
      correlation = if (sd(true_g_c) == 0) NA_real_ else cor(g_median, true_g_c),
      rmse        = sqrt(mean((g_median - true_g_c)^2)),
      sd_true_g   = sd(true_g_c),
      usable      = ok, row.names = NULL
    )

    message("[", label, " T=", T_sim, "] replication ", r, " done, divergences = ", ndiv, ", usable = ", ok)
  }

  reps         <- do.call(rbind, results)
  shape_reps   <- do.call(rbind, shape_results)
  attempts     <- do.call(rbind, attempt_diag)
  div_loc_tbl  <- if (length(div_loc_all) > 0) do.call(rbind, div_loc_all) else NULL

  usable <- reps[reps$usable, ]

  # Loop over the fixed `targets` list, not split()'s dynamically discovered
  # groups (point 3): a scenario/T combination where every replication fails
  # (n_usable = 0) must still produce a row per parameter, with NAs, rather
  # than silently disappearing from the combined summary.
  summary_tbl <- do.call(rbind, lapply(targets, function(p) {
    s <- usable[usable$parameter == p, ]
    n <- nrow(s)
    data.frame(
      scenario        = label,
      T               = T_sim,
      parameter       = p,
      n_usable        = n,
      mean_true_value = if (n) mean(s$true_value) else NA_real_,   # varies by rep for mu/g_sd; see header
      mean_median     = if (n) mean(s$median) else NA_real_,
      bias            = if (n) mean(s$median - s$true_value) else NA_real_,
      rmse            = if (n) sqrt(mean((s$median - s$true_value)^2)) else NA_real_,
      mean_ci_width   = if (n) mean(s$upper_95 - s$lower_95) else NA_real_,
      # Not a meaningful pass/fail check for g_sd's zero-scenario boundary
      # case (point 1) -- read mean_prob_below_delta/frac_upper_ci_below_delta
      # for that instead.
      coverage_95     = if (n) mean(s$true_value >= s$lower_95 & s$true_value <= s$upper_95) else NA_real_,
      mean_prob_below_delta     = if (n && p == "g_sd") mean(s$prob_below_delta) else NA_real_,
      frac_upper_ci_below_delta = if (n && p == "g_sd") mean(s$upper_95 < G_SD_DELTA) else NA_real_
    )
  }))

  shape_usable <- shape_reps[shape_reps$usable, ]
  n_shape <- nrow(shape_usable)
  shape_summary <- data.frame(
    scenario         = label,
    T                = T_sim,
    n_usable         = n_shape,
    mean_correlation = if (n_shape) mean(shape_usable$correlation, na.rm = TRUE) else NA_real_,
    mean_rmse        = if (n_shape) mean(shape_usable$rmse) else NA_real_,
    mean_sd_true_g   = if (n_shape) mean(shape_usable$sd_true_g) else NA_real_
  )

  attempt_summary <- data.frame(
    scenario               = label,
    T                      = T_sim,
    n_attempted            = R_REP,
    n_usable               = sum(attempts$usable),
    usable_rate            = mean(attempts$usable),
    mean_divergences       = mean(attempts$divergences),
    median_divergences     = median(attempts$divergences),
    n_divergence_fail      = sum(!attempts$divergences_ok),
    n_rhat_fail            = sum(!attempts$rhat_ok),
    n_ess_bulk_fail        = sum(!attempts$ess_bulk_ok),
    n_ess_tail_fail        = sum(!attempts$ess_tail_ok),
    n_ebfmi_fail           = sum(!attempts$ebfmi_ok),
    n_lp_fail              = sum(!attempts$lp_ok),
    mean_max_treedepth_hit = mean(attempts$max_treedepth_hit)
  )

  list(
    reps = reps, shape_reps = shape_reps, attempts = attempts, div_loc = div_loc_tbl,
    summary = summary_tbl, shape_summary = shape_summary, attempt_summary = attempt_summary
  )
}

# Writes every component of one run_scenario_replications() result under a
# common file-name suffix.
write_replication_outputs <- function(out, suffix) {
  write.csv(out$reps,            paste0("nnsv_recovery_replications_", suffix, ".csv"), row.names = FALSE)
  write.csv(out$shape_reps,      paste0("nnsv_recovery_shape_", suffix, ".csv"), row.names = FALSE)
  write.csv(out$attempts,        paste0("nnsv_recovery_attempts_", suffix, ".csv"), row.names = FALSE)
  if (!is.null(out$div_loc)) {
    write.csv(out$div_loc,       paste0("nnsv_divergence_diagnostics_", suffix, ".csv"), row.names = FALSE)
  }
  write.csv(out$summary,         paste0("nnsv_recovery_summary_", suffix, ".csv"), row.names = FALSE)
  write.csv(out$shape_summary,   paste0("nnsv_recovery_shape_summary_", suffix, ".csv"), row.names = FALSE)
  write.csv(out$attempt_summary, paste0("nnsv_recovery_attempt_summary_", suffix, ".csv"), row.names = FALSE)
}

combine_field <- function(out_list, field) do.call(rbind, lapply(out_list, `[[`, field))

# =============================================================================
# Part 1: zero/main/strong scenarios at one T
#
# T = 500 here was never chosen to match anything real -- it was
# simulate_nnsv()'s original function default from before this became a
# 3-scenario study. Part 2 below uses the advisor's actual three sizes
# (312/520/1000); once that confirms 520 behaves reasonably, consider
# switching this section to T = 520 too so "main" means the same thing
# everywhere.
# =============================================================================
if (RUN_SCENARIOS) {
  MAIN_T <- 500

  all_out <- lapply(names(scenarios), function(nm) {
    s <- scenarios[[nm]]
    run_scenario_replications(nm, MAIN_T, s$g_scale, s$lev_scale, s$seed_offset)
  })
  names(all_out) <- names(scenarios)

  for (nm in names(scenarios)) write_replication_outputs(all_out[[nm]], nm)

  write.csv(combine_field(all_out, "summary"),         "nnsv_recovery_summary_all.csv", row.names = FALSE)
  write.csv(combine_field(all_out, "shape_summary"),    "nnsv_recovery_shape_summary_all.csv", row.names = FALSE)
  write.csv(combine_field(all_out, "attempt_summary"),  "nnsv_recovery_attempt_summary_all.csv", row.names = FALSE)
  print(combine_field(all_out, "summary"))
  print(combine_field(all_out, "attempt_summary"))

  # Read it as: coverage_95 near 0.95 with small bias = recovery (except g_sd
  # in the zero scenario -- see point 1 in the header, read
  # mean_prob_below_delta/frac_upper_ci_below_delta there instead). Coverage
  # near 0.95 with a large mean_ci_width = the model is honest but the series
  # is too short to be informative; that is a statement about T, not a bug.
  # Compare g_sd's bias/rmse for strong against the single prior_predictive_
  # check.R fit (true 0.404, that one fit's median 0.333): if the replicated
  # bias is consistently negative and of similar size, it is a systematic
  # shrinkage-toward-the-prior effect, not a one-off unlucky draw. Always read
  # the attempt_summary alongside it -- n_usable/usable_rate and the rhat/ESS/
  # E-BFMI failure counts say whether these numbers are conditioning away the
  # hard datasets.
}  # RUN_SCENARIOS

# =============================================================================
# Part 2: T sensitivity (advisor request, see file header) -- ALL THREE
# scenarios (main/zero/strong), each across T = 312 / 520 / 1000. 3 scenarios
# x 3 T's x R_REP fits; T = 1000 fits are the slow ones, so this is 9x a
# single-T scenario run and 3x a main-only T run -- expect it to take a
# while.
# =============================================================================
if (RUN_T_SENSITIVITY) {
  T_VALUES <- c(312, 520, 1000)

  t_plan <- expand.grid(
    scenario = names(scenarios), T_sim = T_VALUES,
    stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE
  )

  t_out <- lapply(seq_len(nrow(t_plan)), function(i) {
    nm    <- t_plan$scenario[i]
    T_sim <- t_plan$T_sim[i]
    s <- scenarios[[nm]]
    run_scenario_replications(
      label = nm, T_sim = T_sim,
      g_scale = s$g_scale, lev_scale = s$lev_scale,
      # offset by T so seeds don't collide across the three sizes, the three
      # scenarios, or Part 1's own T=500 run of the same scenarios
      seed_offset = s$seed_offset + T_sim * 10L
    )
  })
  names(t_out) <- paste0(t_plan$scenario, "_T", t_plan$T_sim)

  for (nm in names(t_out)) write_replication_outputs(t_out[[nm]], nm)

  t_summary <- combine_field(t_out, "summary")
  t_summary <- t_summary[order(t_summary$scenario, t_summary$parameter, t_summary$T), ]
  write.csv(t_summary, "nnsv_recovery_summary_by_T.csv", row.names = FALSE)
  print(t_summary)

  t_shape_summary <- combine_field(t_out, "shape_summary")
  t_shape_summary <- t_shape_summary[order(t_shape_summary$scenario, t_shape_summary$T), ]
  write.csv(t_shape_summary, "nnsv_recovery_shape_summary_by_T.csv", row.names = FALSE)
  print(t_shape_summary)

  t_attempt_summary <- combine_field(t_out, "attempt_summary")
  t_attempt_summary <- t_attempt_summary[order(t_attempt_summary$scenario, t_attempt_summary$T), ]
  write.csv(t_attempt_summary, "nnsv_recovery_attempt_summary_by_T.csv", row.names = FALSE)
  print(t_attempt_summary)

  # Read across T within each (scenario, parameter), especially g_sd: bias
  # shrinking and coverage_95 moving toward 0.95 as T grows from 312 to 520 to
  # 1000 is the "more data helps, and 520 already captures most of the
  # benefit" argument -- for main this settles "why T = 520"; for zero/strong
  # it additionally shows whether the false-positive control and the
  # identifiability stress test both hold up, or only kick in at larger T. If
  # 520 and 1000 look almost identical while 312 is clearly worse, that is the
  # strongest version of "T = 520 was not an accident" the advisor asked for.
  # If 1000 keeps improving a lot over 520, say that honestly too -- a real
  # limitation of the sample, not a reason to hide the T = 1000 row. Cross-
  # check every number here against nnsv_recovery_attempt_summary_by_T.csv's
  # usable_rate before writing any of it up.
}  # RUN_T_SENSITIVITY

# =============================================================================
# Part 3 (2026-08-23): advisor's two fixes together, T = 520 (the real sample
# size), all three scenarios, R_REP replications each.
#
#   Fix 1: s_fixed is now 1.0 (see stan_data above), not 0.50. The strong
#   scenario's g_sd was consistently underestimated with s_fixed = 0.50
#   because |g| <= s_fixed is a hard bound, and the true g often exceeds 0.5
#   when true g_sd ~ 0.5 -- the model literally cannot represent it. Expect
#   that bias to shrink a lot here.
#
#   Fix 2: the usable gate is now NEW_GATE_PARS (g_sd, h_bar, phi, sigma_eta,
#   nu, lp__), not mu/tau_w/raw weights. Checked against the existing T = 312/
#   520/1000 attempts CSVs (s_fixed = 0.50): every usable == FALSE case there
#   was divergence-driven, not rhat/ess-driven, so this narrower gate would
#   not have changed those numbers -- but if s_fixed = 1.0 also reduces
#   divergences (plausible: the old bound may have been straining the
#   network's geometry, not just capping its output), usable_rate here could
#   genuinely be higher than the old T = 520 rows, for a real reason this
#   time.
#
# Suffixed "_s1_T520" throughout so nothing overwrites the existing s = 0.50
# T-sensitivity files -- the two are meant to be compared side by side.
# =============================================================================
if (RUN_S1_FIX) {
  S1_FIX_T <- 520

  s1_out <- lapply(names(scenarios), function(nm) {
    s <- scenarios[[nm]]
    run_scenario_replications(
      label = nm, T_sim = S1_FIX_T,
      g_scale = s$g_scale, lev_scale = s$lev_scale,
      # offset so seeds don't collide with Part 2's own T = 520 run of the
      # same scenarios (s_fixed = 0.50)
      seed_offset = s$seed_offset + S1_FIX_T * 100L
    )
  })
  names(s1_out) <- paste0(names(scenarios), "_s1_T", S1_FIX_T)

  for (nm in names(s1_out)) write_replication_outputs(s1_out[[nm]], nm)

  s1_summary <- combine_field(s1_out, "summary")
  s1_summary <- s1_summary[order(s1_summary$scenario, s1_summary$parameter), ]
  write.csv(s1_summary, "nnsv_recovery_summary_s1_T520.csv", row.names = FALSE)
  print(s1_summary)

  s1_shape_summary <- combine_field(s1_out, "shape_summary")
  write.csv(s1_shape_summary, "nnsv_recovery_shape_summary_s1_T520.csv", row.names = FALSE)
  print(s1_shape_summary)

  s1_attempt_summary <- combine_field(s1_out, "attempt_summary")
  write.csv(s1_attempt_summary, "nnsv_recovery_attempt_summary_s1_T520.csv", row.names = FALSE)
  print(s1_attempt_summary)

  # Compare directly against the T = 520 rows of nnsv_recovery_summary_by_T.csv
  # / nnsv_recovery_attempt_summary_by_T.csv (s_fixed = 0.50, old gate):
  #   - strong's g_sd bias should shrink a lot (Fix 1 working).
  #   - usable_rate may or may not move; if it does, that's Fix 1 fixing
  #     divergences as a side effect, not Fix 2 (Fix 2 was shown to be a
  #     no-op on the old data).
}  # RUN_S1_FIX
