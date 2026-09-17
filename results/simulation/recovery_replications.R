# recovery_replications.R

library(rstan)
library(posterior)

rstan_options(auto_write = TRUE)
options(benchmark.nn_autorun = FALSE)
source("../../nonlinear_sv.R", chdir = TRUE)

options(mc.cores = 4)
rstan_options(auto_write = TRUE)
rstan_options(threads_per_chain = 1) 
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
      prior_only = 0L,        
      stationary_init = 0L,   
      h1_scale = 2.0       
    )

    linear_start <- nn_sv_linear_start(stan_data$y, stan_data$X)
    init_fn <- function() nn_sv_init(stan_data, previous_post = NULL, linear_start = linear_start)

    fit <- sampling(
      model, data = stan_data, chains = 4, cores = 4,
      iter = 3000, warmup = 1500, seed = 1000 + r, refresh = 0, init = init_fn,
      control = list(adapt_delta = 0.99, max_treedepth = 12)
    )

    # -------------------------------------------------------------------------
    # Full attempt-level diagnostics
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
    # Shape of g
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
      mean_true_value = if (n) mean(s$true_value) else NA_real_,   # varies by rep for mu/g_sd
      mean_median     = if (n) mean(s$median) else NA_real_,
      bias            = if (n) mean(s$median - s$true_value) else NA_real_,
      rmse            = if (n) sqrt(mean((s$median - s$true_value)^2)) else NA_real_,
      mean_ci_width   = if (n) mean(s$upper_95 - s$lower_95) else NA_real_,
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
# T = 520 here was never chosen to match anything real -- it was
# simulate_nnsv()'s original function default from before this became a
# 3-scenario study. Part 2 below uses the advisor's actual three sizes
# (312/520/1000); once that confirms 520 behaves reasonably, consider
# switching this section to T = 520 too so "main" means the same thing
# everywhere.
# =============================================================================
if (RUN_SCENARIOS) {
  MAIN_T <- 520

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

}  # RUN_SCENARIOS

# =============================================================================
# Part 2: T sensitivity-- ALL THREE
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
      # scenarios, or Part 1's own T=520 run of the same scenarios
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
}  # RUN_T_SENSITIVITY

# =============================================================================
# Part 3 : T = 520 (the real sample
# size), all three scenarios, R_REP replications each.
#
#   Fix 1: s_fixed is now 1.0 (see stan_data above), not 0.50.
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

}  # RUN_S1_FIX
