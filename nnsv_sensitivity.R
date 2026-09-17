
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
options(benchmark.nn_autorun = FALSE)
source("nonlinear_sv.R")
rstan_options(auto_write = TRUE)



SENS_SIM_DIR    <- "results/simulation"
SENS_OUTPUT_DIR <- "results/nnsv_sensitivity"

SENS_SCENARIOS    <- c("main", "strong")
SENS_K_VALUES     <- 4L    
SENS_TAU_W_SCALES <- c(0.2, 0.5, 1.0)
SENS_S_FIXED      <- 1.0   


SENS_CHAINS        <- 4L
SENS_WARMUP        <- 2000L
SENS_SAMPLING      <- 4000L
SENS_ADAPT_DELTA   <- 0.99
SENS_MAX_TREEDEPTH <- 12L
SENS_FIT_SEED      <- 20260820L

sens_targets <- c("mu", "phi", "sigma_eta", "nu", "g_sd", "h_bar")


# ---------------------------------------------------------------------------
# one senario
# ---------------------------------------------------------------------------
load_sim_scenario <- function(label, sim_dir = SENS_SIM_DIR) {
  sim_data <- utils::read.csv(file.path(sim_dir, sprintf("nnsv_simulation_%s.csv", label)))
  T_sim <- nrow(sim_data)
  g <- sim_data$g_true[2:T_sim]

  true_params_path <- file.path(sim_dir, sprintf("nnsv_%s_true_parameters.csv", label))
  if (file.exists(true_params_path)) {
    tp <- utils::read.csv(true_params_path)
    get_true <- function(nm) tp$true_value[tp$parameter == nm]
    true_vec <- c(mu = get_true("mu"), phi = get_true("phi"),
                 sigma_eta = get_true("sigma_eta"), nu = get_true("nu"),
                 g_sd = get_true("g_sd"))
  } else {
    true_vec <- c(
      mu        = 1.80 + mean(g) / (1 - 0.95),
      phi       = 0.95,
      sigma_eta = 0.25,
      nu        = 10,
      g_sd      = sd(g)
    )
  }
  true_vec["h_bar"] <- mean(sim_data$h_true)

  list(sim_data = sim_data, T_sim = T_sim, true_vec = true_vec)
}

run_sens_cell <- function(compiled_model, label, scenario, K, tau_w_scale, seed) {
  sim_data <- scenario$sim_data
  T_sim    <- scenario$T_sim
  true_vec <- scenario$true_vec

  stan_data <- list(
    T = T_sim, D = 1, K = K,
    y = sim_data$y,
    X = matrix(sim_data$x[1:(T_sim - 1)], ncol = 1),
    use_student_t = 1,
    x_forecast = array(sim_data$x[T_sim], dim = 1),
    s_fixed = SENS_S_FIXED, tau_w_scale = tau_w_scale,
    mu_scale = 5, phi_a = 20, phi_b = 1.5,
    sigma_eta_scale = 1, b1_scale = 0.5, nu_rate = 0.1,
    prior_only = 0L, stationary_init = 0L, h1_scale = 2.0
  )

  linear_start <- nn_sv_linear_start(stan_data$y, stan_data$X)
  init_fn <- function() nn_sv_init(stan_data, previous_post = NULL, linear_start = linear_start)

  fit <- rstan::sampling(
    compiled_model, data = stan_data, chains = SENS_CHAINS, cores = SENS_CHAINS,
    iter = SENS_WARMUP + SENS_SAMPLING, warmup = SENS_WARMUP, seed = seed,
    refresh = 0, init = init_fn,
    control = list(adapt_delta = SENS_ADAPT_DELTA, max_treedepth = SENS_MAX_TREEDEPTH)
  )

  diagnostics <- stan_fit_diagnostics(fit, pars = c(sens_targets, "tau_w"))

  dr <- rstan::extract(fit, pars = sens_targets)
  recovery <- do.call(rbind, lapply(sens_targets, function(p) {
    v   <- as.numeric(dr[[p]])
    med <- median(v); s <- sd(v)
    lo  <- quantile(v, 0.025, names = FALSE); hi <- quantile(v, 0.975, names = FALSE)
    tv  <- unname(true_vec[p])
    data.frame(
      scenario = label, K = K, tau_w_scale = tau_w_scale,
      parameter = p, true_value = tv,
      posterior_median = med, posterior_sd = s, lower_95 = lo, upper_95 = hi,
      inside_95 = tv >= lo & tv <= hi,
      bias      = med - tv,
      bias_frac = if (tv != 0) (med - tv) / tv else NA_real_,
      z         = (med - tv) / s
    )
  }))

  true_g_c <- sim_data$g_true[2:T_sim] - mean(sim_data$g_true[2:T_sim])
  g_draws  <- rstan::extract(fit, pars = "g")$g
  g_median <- apply(g_draws, 2, median)
  shape <- data.frame(
    scenario = label, K = K, tau_w_scale = tau_w_scale,
    correlation = if (sd(true_g_c) == 0) NA_real_ else cor(g_median, true_g_c),
    rmse        = sqrt(mean((g_median - true_g_c)^2))
  )

  attempt <- data.frame(
    scenario = label, K = K, tau_w_scale = tau_w_scale,
    divergences  = diagnostics$divergence_sum,
    max_rhat     = diagnostics$max_rhat,
    min_ess_bulk = diagnostics$min_bulk_ess,
    min_ess_tail = diagnostics$min_tail_ess,
    min_ebfmi    = diagnostics$min_ebfmi,
    passes_gate  = isTRUE(diagnostics$all_checks_passed)
  )

  message(sprintf("[%s K=%d tw=%.2f s=%.2f] divergences=%d, passes_gate=%s",
                  label, K, tau_w_scale, SENS_S_FIXED, diagnostics$divergence_sum,
                  isTRUE(diagnostics$all_checks_passed)))
  rm(fit); invisible(gc())

  list(recovery = recovery, shape = shape, attempt = attempt)
}

# ---------------------------------------------------------------------------
# full grid
# ---------------------------------------------------------------------------
run_sens_grid <- function(compiled_model,
                          scenarios = SENS_SCENARIOS,
                          K_values = SENS_K_VALUES,
                          tau_w_scales = SENS_TAU_W_SCALES,
                          output_dir = SENS_OUTPUT_DIR) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  loaded <- stats::setNames(lapply(scenarios, load_sim_scenario), scenarios)

  plan <- expand.grid(
    scenario = scenarios, K = K_values, tau_w_scale = tau_w_scales,
    stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE
  )

  cells <- lapply(seq_len(nrow(plan)), function(i) {
    run_sens_cell(
      compiled_model = compiled_model,
      label = plan$scenario[i], scenario = loaded[[plan$scenario[i]]],
      K = plan$K[i], tau_w_scale = plan$tau_w_scale[i],
      seed = SENS_FIT_SEED + i - 1L
    )
  })

  recovery_tbl <- do.call(rbind, lapply(cells, `[[`, "recovery"))
  shape_tbl    <- do.call(rbind, lapply(cells, `[[`, "shape"))
  attempt_tbl  <- do.call(rbind, lapply(cells, `[[`, "attempt"))

  write.csv(recovery_tbl, file.path(output_dir, "ks_sensitivity_recovery.csv"), row.names = FALSE)
  write.csv(shape_tbl,    file.path(output_dir, "ks_sensitivity_shape.csv"), row.names = FALSE)
  write.csv(attempt_tbl,  file.path(output_dir, "ks_sensitivity_attempts.csv"), row.names = FALSE)

  g_sd_rows <- recovery_tbl[recovery_tbl$parameter == "g_sd",
                            c("scenario", "K", "tau_w_scale",
                             "true_value", "posterior_median", "bias", "bias_frac", "inside_95")]
  shrinkage_tbl <- merge(g_sd_rows, attempt_tbl, by = c("scenario", "K", "tau_w_scale"))
  shrinkage_tbl <- merge(shrinkage_tbl,
                         shape_tbl[, c("scenario", "K", "tau_w_scale", "correlation", "rmse")],
                         by = c("scenario", "K", "tau_w_scale"))
  shrinkage_tbl <- shrinkage_tbl[order(shrinkage_tbl$scenario, shrinkage_tbl$K,
                                       shrinkage_tbl$tau_w_scale), ]
  write.csv(shrinkage_tbl, file.path(output_dir, "ks_shrinkage_readout.csv"), row.names = FALSE)
  print(shrinkage_tbl)

  list(recovery = recovery_tbl, shape = shape_tbl, attempts = attempt_tbl,
       shrinkage_readout = shrinkage_tbl)
}


# ==============================================================================
# main
# ==============================================================================

if (isTRUE(getOption("benchmark.nn_sensitivity_autorun", TRUE))) {
  config <- nn_sv_default_config()
  config$chains <- SENS_CHAINS
  configure_nn_sv_parallel(config$chains)
  compiled_model <- rstan::stan_model(file = config$stan_file)

  nnsv_sensitivity_result <- run_sens_grid(compiled_model = compiled_model)
}
