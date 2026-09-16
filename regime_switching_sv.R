if (!exists("rolling_forecast", mode = "function", inherits = TRUE)) {
  source("benchmark_utils.R")
}

# 2026-08-25: mu_scale/phi_a/phi_b/nu_rate default to the same values as
# linear_sv_stan_data()/nn_sv_default_config() (5, 20, 1.5, 0.1) -- see
# regime_switching_sv.stan's data block comment. sigma_eta has no analogous
# argument here: its prior stays hardcoded in the .stan file on purpose.
ms_sv_stan_data <- function(bench, train_indices,
                            stationary_init = 0L, h1_scale = 2.0,
                            mu_scale = 5, phi_a = 20, phi_b = 1.5,
                            nu_rate = 0.1) {
  list(
    T = length(train_indices),
    D = ncol(bench$X_all),
    y = bench$y_all[train_indices],
    X = bench$X_all[train_indices, , drop = FALSE],
    stationary_init = as.integer(as.logical(stationary_init)),
    h1_scale = as.numeric(h1_scale),
    mu_scale = as.numeric(mu_scale),
    phi_a = as.numeric(phi_a),
    phi_b = as.numeric(phi_b),
    nu_rate = as.numeric(nu_rate)
  )
}

ms_sv_initial_h <- function(y) {
  squared_returns <- as.numeric(y)^2
  finite_squared_returns <- squared_returns[is.finite(squared_returns)]
  if (!length(finite_squared_returns)) {
    warning("MS-SV initialization received no finite returns; using a flat initial state.",
            call. = FALSE)
    return(rep(0, length(squared_returns)))
  }
  variance_floor <- max(
    stats::quantile(finite_squared_returns, probs = 0.05, na.rm = TRUE) / 10,
    1e-4
  )
  log(pmax(squared_returns, variance_floor))
}

ms_sv_phi_to_unit <- function(phi) {
  pmin(pmax(phi / 0.995, 1e-6), 1 - 1e-6)
}

ms_sv_probability_to_unit <- function(probability) {
  pmin(pmax((probability - 0.005) / 0.99, 1e-6), 1 - 1e-6)
}

init_ms_sv <- function(stan_data, previous_post = NULL) {
  if (is.null(previous_post)) {
    mu <- sort(c(-1, 1) + log(mean(stan_data$y^2, na.rm = TRUE)))
    return(list(
      mu = mu,
      phi_unit = ms_sv_phi_to_unit(0.95),
      log_sigma_eta = log(0.25),
      beta = rep(0, stan_data$D),
      eta_raw = matrix(
        stats::rnorm(stan_data$T * 2L, mean = 0, sd = 0.1),
        nrow = stan_data$T,
        ncol = 2L
      ),
      p11_unit = ms_sv_probability_to_unit(0.95),
      p22_unit = ms_sv_probability_to_unit(0.95),
      nu_minus2 = 8
    ))
  }

  mu <- sort(apply(previous_post$mu, 2, stats::median))
  if (diff(mu) < 1e-4) mu[2] <- mu[1] + 0.1
  list(
    mu = mu,
    phi_unit = ms_sv_phi_to_unit(stats::median(previous_post$phi)),
    log_sigma_eta = log(max(stats::median(previous_post$sigma_eta), 1e-4)),
    beta = apply(previous_post$beta, 2, stats::median),
    eta_raw = matrix(
      stats::rnorm(stan_data$T * 2L, mean = 0, sd = 0.1),
      nrow = stan_data$T,
      ncol = 2L
    ),
    p11_unit = ms_sv_probability_to_unit(stats::median(previous_post$p11)),
    p22_unit = ms_sv_probability_to_unit(stats::median(previous_post$p22)),
    nu_minus2 = max(stats::median(previous_post$nu_minus2), 1e-4)
  )
}

fit_regime_switching_sv_refit <- function(compiled_model,
                                          train_indices,
                                          previous_model,
                                          refit_id,
                                          bench,
                                          chains,
                                          iter,
                                          warmup,
                                          seed,
                                          adapt_delta,
                                          max_treedepth,
                                          stationary_init = 0L,
                                          h1_scale = 2.0) {
  stan_data <- ms_sv_stan_data(bench, train_indices, stationary_init, h1_scale)
  previous_post <- if (is.null(previous_model)) NULL else previous_model$posterior

  fit <- rstan::sampling(
    object = compiled_model,
    data = stan_data,
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = as.integer(seed) + as.integer(refit_id) - 1L,
    init = function() init_ms_sv(stan_data, previous_post),
    control = list(
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth
    )
  )
  post <- rstan::extract(fit)
  n_draws <- length(post$p11)
  last_t <- length(train_indices)
  p_regime_1 <- post$filtered_prob_last[, 1]

  # 2026-08-24: judge convergence only on the quantities actually used
  # downstream, same rule as linear/nonlinear SV. MS-SV has no h_bar
  # analogue, so mu (both regime components) stays in the set rather than
  # being replaced; beta and eta_raw (the "raw weights" here -- inputs the
  # forecast never reads directly) and the unit-scale nuisance
  # reparameterisations (phi_unit/log_sigma_eta/p11_unit/p22_unit) are
  # dropped in favour of the natural-scale quantities forecast_ms_sv_one_step()
  # actually consumes (phi, sigma_eta, p11, p22), plus nu and the
  # log-likelihood.
  ms_diagnostic_pars <- c("mu", "phi", "sigma_eta", "nu", "p11", "p22", "lp__")
  diagnostics <- stan_fit_diagnostics(fit, pars = ms_diagnostic_pars)
  assert_stan_diagnostics(
    diagnostics = diagnostics,
    model_name = sprintf("Regime-switching SV refit %d", refit_id),
    strict = FALSE
  )

  list(
    model = list(posterior = post),
    state = list(
      h_paths = matrix(
        post$h[, last_t, , drop = FALSE],
        nrow = n_draws,
        ncol = 2L
      ),
      regime = ifelse(stats::rbinom(n_draws, 1, p_regime_1) == 1L, 1L, 2L),
      draw_index = seq_len(n_draws)
    ),
    fit = fit,
    diagnostics = diagnostics,
    usable = isTRUE(diagnostics$all_checks_passed)
  )
}

# 2026-08-24: fits only the initial training window and reports its
# convergence -- a deliberate, manual first step (not an automatic gate any
# more): run this via PREFLIGHT_ONLY in run_models.R, look at the real
# diagnostics (with CHECK_CONVERGENCE = TRUE), and only then decide whether
# to flip PREFLIGHT_ONLY off and let run_regime_switching_sv() below roll
# through validation/test.
run_regime_switching_sv_preflight <- function(compiled_model,
                                              bench,
                                              chains,
                                              iter,
                                              warmup,
                                              seed,
                                              adapt_delta,
                                              max_treedepth,
                                              stationary_init = 0L,
                                              h1_scale = 2.0) {
  train_indices <- bench$split_rows$train
  fitted <- fit_regime_switching_sv_refit(
    compiled_model = compiled_model,
    train_indices = train_indices,
    previous_model = NULL,
    refit_id = 1L,
    bench = bench,
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = seed,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth,
    stationary_init = stationary_init,
    h1_scale = h1_scale
  )

  passed <- isTRUE(fitted$usable)
  list(
    passed = passed,
    fitted = fitted,
    diagnostics = fitted$diagnostics,
    train_indices = train_indices,
    message = if (passed) {
      "Initial regime-switching SV Stan diagnostic check (training window only) passed."
    } else {
      "Initial regime-switching SV Stan diagnostic check (training window only) failed; validation/test were not run."
    }
  )
}

make_regime_switching_sv_fitter <- function(compiled_model, chains, iter,
                                            warmup, seed, adapt_delta,
                                            max_treedepth,
                                            first_refit = NULL,
                                            stationary_init = 0L,
                                            h1_scale = 2.0) {
  function(train_indices, previous_model, refit_id, bench) {
    if (identical(as.integer(refit_id), 1L) && !is.null(first_refit)) {
      return(first_refit)
    }
    fit_regime_switching_sv_refit(
      compiled_model, train_indices, previous_model, refit_id, bench,
      chains, iter, warmup, seed, adapt_delta, max_treedepth,
      stationary_init, h1_scale
    )
  }
}

forecast_ms_sv_one_step <- function(model, state, forecast_index, bench) {
  post <- model$posterior
  draw_index <- state$draw_index
  n_draws <- length(draw_index)
  staying_probability <- ifelse(
    state$regime == 1L,
    post$p11[draw_index],
    post$p22[draw_index]
  )
  next_regime <- ifelse(
    stats::rbinom(n_draws, 1, staying_probability) == 1L,
    state$regime,
    3L - state$regime
  )
  x_t <- bench$X_all[forecast_index, ]
  x_effect <- as.vector(post$beta[draw_index, , drop = FALSE] %*% x_t)
  h_paths_t <- matrix(NA_real_, nrow = n_draws, ncol = 2L)
  for (s in 1:2) {
    parameter_index <- cbind(draw_index, rep.int(s, n_draws))
    transition_mean <- post$mu[parameter_index] +
      post$phi[draw_index] * (
        state$h_paths[, s] - post$mu[parameter_index]
      ) + x_effect
    h_paths_t[, s] <- transition_mean +
      post$sigma_eta[draw_index] * stats::rnorm(n_draws)
  }
  h_t <- h_paths_t[cbind(seq_len(n_draws), next_regime)]
  nu_t <- post$nu_minus2[draw_index] + 2
  scale_t <- sqrt(exp(h_t))

  list(
    variance_draws = exp(h_t) * nu_t / (nu_t - 2),
    return_draws = stats::rt(n_draws, df = nu_t) * scale_t,
    state_prior = list(
      h_paths = h_paths_t,
      regime = next_regime,
      draw_index = draw_index
    ),
    log_density_draws = function(y) {
      stats::dt(y / scale_t, df = nu_t, log = TRUE) - log(scale_t)
    }
  )
}

update_ms_sv_state <- function(model, state, forecast, observed_y,
                               observation_index, bench) {
  log_weights <- forecast$log_density_draws(observed_y)
  weights <- exp(log_weights - max(log_weights))
  selected <- sample.int(length(weights), length(weights), replace = TRUE,
                         prob = weights)
  list(
    h_paths = forecast$state_prior$h_paths[selected, , drop = FALSE],
    regime = forecast$state_prior$regime[selected],
    draw_index = forecast$state_prior$draw_index[selected]
  )
}

run_regime_switching_sv <- function(
    bench = load_benchmark_data(),
    refit_every = getOption("benchmark.refit_every", 13L),
    save_fits = getOption("benchmark.save_fits", FALSE),
    seed = 1666L,
    chains = getOption("benchmark.stan_chains", 4L),
    iter = getOption("benchmark.stan_iter", 4000L),
    warmup = getOption("benchmark.stan_warmup", 2000L),
    adapt_delta = getOption("benchmark.ms_stan_adapt_delta", 0.99),
    max_treedepth = getOption("benchmark.stan_max_treedepth", 10L),
    # See regime_switching_sv.stan: same phi -> 1 funnel nonlinear_sv.stan
    # had before its 2026-08-21 fix, same shaped phi prior pushing toward it.
    stationary_init = getOption("benchmark.ms_stationary_init", 0L),
    h1_scale = getOption("benchmark.ms_h1_scale", 2.0)) {
  rstan::rstan_options(auto_write = TRUE)
  cores <- parallel::detectCores(logical = TRUE)
  if (is.na(cores)) cores <- 1L
  options(mc.cores = min(as.integer(chains), cores))
  compiled_model <- rstan::stan_model(file = "regime_switching_sv.stan")

  # No preflight gate any more (2026-08-24): see run_models.R's
  # PREFLIGHT_ONLY comment. First refit is fit directly inside
  # rolling_forecast() below rather than pre-checked and reused.
  result <- rolling_forecast(
    bench = bench,
    fit_model = make_regime_switching_sv_fitter(
      compiled_model, chains, iter, warmup, seed, adapt_delta,
      max_treedepth, NULL, stationary_init, h1_scale
    ),
    forecast_one_step = forecast_ms_sv_one_step,
    update_state = update_ms_sv_state,
    refit_every = refit_every,
    model_name = "regimeSwitchingSV",
    save_fits = save_fits,
    seed = seed
  )
  result$posterior <- result$model$posterior
  result
}

if (isTRUE(getOption("benchmark.ms_sv_autorun", FALSE))) {
  bench <- load_benchmark_data()
  if (!exists("benchmark_results")) benchmark_results <- list()
  benchmark_results$regimeSwitchingSV <- run_regime_switching_sv(bench = bench)
}
