if (!exists("rolling_forecast", mode = "function", inherits = TRUE)) {
  source("benchmark_utils.R")
}

linear_sv_stan_data <- function(bench, train_indices,
                                use_student_t = TRUE,
                                prior_only = 0L,
                                stationary_init = 0L,
                                h1_scale = 2.0,
                                # Same priors as nonlinear_sv_v3.stan's defaults
                                # throughout the simulation study, so the two
                                # models are compared on equal footing.
                                mu_scale = 5,
                                phi_a = 20,
                                phi_b = 1.5,
                                sigma_eta_scale = 1,
                                nu_rate = 0.1) {
  T_train <- length(train_indices)
  forecast_row <- max(train_indices) + 1L
  x_forecast <- if (forecast_row <= nrow(bench$X_all)) {
    bench$X_all[forecast_row, ]
  } else {
    bench$X_all[max(train_indices), ]
  }
  list(
    T = T_train,
    D = ncol(bench$X_all),
    K = 1L,                                  
    y = bench$y_all[train_indices],
    X = bench$X_all[train_indices, , drop = FALSE],
    use_student_t = as.integer(isTRUE(use_student_t)),
    x_forecast = as.numeric(x_forecast),
    s_fixed = 1,                            
    tau_w_scale = 0.2,                         
    mu_scale = as.numeric(mu_scale),
    phi_a = as.numeric(phi_a),
    phi_b = as.numeric(phi_b),
    sigma_eta_scale = as.numeric(sigma_eta_scale),
    b1_scale = 0.5,                          
    nu_rate = as.numeric(nu_rate),
    prior_only = as.integer(as.logical(prior_only)),
    stationary_init = as.integer(as.logical(stationary_init)),
    h1_scale = as.numeric(h1_scale)
  )
}

# Cheap AR(1)-on-log-squared-returns warm start for mu/phi/sigma_eta only --
# the same trick as nonlinear_sv.R's nn_sv_linear_start(). beta starts at zero
# instead (linear_sv_init() below), same convention as regime_switching_sv.R's
# init_ms_sv() cold start. Duplicated rather than shared so linear_sv.R
# doesn't need nonlinear_sv.R sourced to run standalone.
linear_sv_warm_start <- function(y) {
  z <- log(pmax(y^2, .Machine$double.eps)) + 1.2704
  mu_start <- mean(z, na.rm = TRUE)
  phi_start <- 0.95
  sigma_start <- 0.25

  if (length(z) >= 10L) {
    fit <- tryCatch(
      stats::lm(z_now ~ z_lag, data = data.frame(z_now = z[-1L], z_lag = z[-length(z)])),
      error = function(e) NULL
    )
    if (!is.null(fit)) {
      coefficients <- stats::coef(fit)
      if (is.finite(coefficients["z_lag"])) {
        phi_start <- min(max(unname(coefficients["z_lag"]), 0.50), 0.98)
      }
      residual_sd <- stats::sd(stats::residuals(fit), na.rm = TRUE)
      if (is.finite(residual_sd)) sigma_start <- min(max(residual_sd, 0.05), 1.00)
    }
  }
  list(mu = mu_start, phi = phi_start, sigma_eta = sigma_start)
}

linear_sv_init <- function(stan_data, previous_post = NULL, warm_start = NULL) {
  if (is.null(previous_post)) {
    start <- if (is.null(warm_start)) linear_sv_warm_start(stan_data$y) else warm_start
    values <- list(
      mu        = start$mu,
      phi_star  = (start$phi + 1) / 2,
      sigma_eta = start$sigma_eta,
      beta      = rep(0, stan_data$D),
      eta_raw   = rep(0, stan_data$T)
    )
  } else {
    values <- list(
      mu        = stats::median(previous_post$mu),
      phi_star  = (stats::median(previous_post$phi) + 1) / 2,
      sigma_eta = max(stats::median(previous_post$sigma_eta), 1e-4),
      beta      = apply(previous_post$beta, 2, stats::median),
      eta_raw   = rep(0, stan_data$T)
    )
  }
  if (stan_data$use_student_t == 1L) {
    nu_minus2_start <- if (is.null(previous_post)) 8 else
      max(stats::median(previous_post$nu_minus2), 1e-4)
    values$nu_minus2 <- array(nu_minus2_start, dim = 1L)
  }
  values
}

fit_linear_sv_refit <- function(compiled_model,
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
  stan_data <- linear_sv_stan_data(
    bench, train_indices,
    stationary_init = stationary_init, h1_scale = h1_scale
  )
  previous_post <- if (is.null(previous_model)) NULL else previous_model$posterior

  fit <- rstan::sampling(
    object = compiled_model,
    data = stan_data,
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = as.integer(seed) + as.integer(refit_id) - 1L,
    init = function() linear_sv_init(stan_data, previous_post),
    control = list(
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth
    )
  )
  post <- rstan::extract(fit)
  n_draws <- length(post$mu)

  linear_diagnostic_pars <- c("h_bar", "phi", "sigma_eta", "nu", "lp__")
  diagnostics <- stan_fit_diagnostics(fit, pars = linear_diagnostic_pars)
  assert_stan_diagnostics(
    diagnostics = diagnostics,
    model_name = "Linear SV",
    strict = FALSE
  )

  list(
    model = list(posterior = post),
    state = list(
      h = post$h[, length(train_indices)],
      draw_index = seq_len(n_draws)
    ),
    fit = fit,
    diagnostics = diagnostics,
    usable = isTRUE(diagnostics$all_checks_passed)
  )
}

run_linear_sv_preflight <- function(compiled_model,
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
  fitted <- fit_linear_sv_refit(
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
      "Initial Linear SV Stan diagnostic check (training window only) passed."
    } else {
      "Initial Linear SV Stan diagnostic check (training window only) failed; validation/test were not run."
    }
  )
}

make_linear_sv_fitter <- function(compiled_model, chains, iter, warmup, seed,
                                  adapt_delta, max_treedepth,
                                  first_refit = NULL,
                                  stationary_init = 0L,
                                  h1_scale = 2.0) {
  function(train_indices, previous_model, refit_id, bench) {
    if (identical(as.integer(refit_id), 1L) && !is.null(first_refit)) {
      return(first_refit)
    }
    fit_linear_sv_refit(
      compiled_model, train_indices, previous_model, refit_id, bench,
      chains, iter, warmup, seed, adapt_delta, max_treedepth,
      stationary_init, h1_scale
    )
  }
}


forecast_linear_sv_one_step <- function(model, state, forecast_index, bench) {
  post <- model$posterior
  draw_index <- state$draw_index
  n_draws <- length(draw_index)
  x_t <- bench$X_all[forecast_index, ]
  x_effect <- as.vector(post$beta[draw_index, , drop = FALSE] %*% x_t)

  transition_mean <- post$mu[draw_index] +
    post$phi[draw_index] * (state$h - post$mu[draw_index]) + x_effect
  h_t <- transition_mean + post$sigma_eta[draw_index] * stats::rnorm(n_draws)
  nu_t <- post$nu_minus2[draw_index] + 2
  scale_t <- sqrt(exp(h_t))

  list(
    variance_draws = exp(h_t) * nu_t / (nu_t - 2),
    return_draws = stats::rt(n_draws, df = nu_t) * scale_t,
    state_prior = list(h = h_t, draw_index = draw_index),
    log_density_draws = function(y) {
      stats::dt(y / scale_t, df = nu_t, log = TRUE) - log(scale_t)
    }
  )
}

update_linear_sv_state <- function(model, state, forecast, observed_y,
                                   observation_index, bench) {
  log_weights <- forecast$log_density_draws(observed_y)
  weights <- exp(log_weights - max(log_weights))
  selected <- sample.int(length(weights), length(weights), replace = TRUE,
                         prob = weights)

  list(
    h = forecast$state_prior$h[selected],
    draw_index = forecast$state_prior$draw_index[selected]
  )
}

run_linear_sv <- function(bench = load_benchmark_data(),
                          refit_every = getOption("benchmark.refit_every", 13L),
                          save_fits = getOption("benchmark.save_fits", FALSE),
                          seed = 666L,
                          chains = getOption("benchmark.stan_chains", 4L),
                          iter = getOption("benchmark.stan_iter", 4000L),
                          warmup = getOption("benchmark.stan_warmup", 2000L),
                          adapt_delta = getOption("benchmark.linear_stan_adapt_delta", 0.99),
                          max_treedepth = getOption("benchmark.stan_max_treedepth", 10L),
                          # Same fix as nonlinear_sv_v3.stan / regime_switching_sv.stan:
                          # h[1]'s stationary scale explodes as phi -> 1.
                          stationary_init = getOption("benchmark.linear_stationary_init", 0L),
                          h1_scale = getOption("benchmark.linear_h1_scale", 2.0)) {
  rstan::rstan_options(auto_write = TRUE)
  cores <- parallel::detectCores(logical = TRUE)
  if (is.na(cores)) cores <- 1L
  options(mc.cores = min(as.integer(chains), cores))

  compiled_model <- rstan::stan_model(file = "linear_sv.stan")


  result <- rolling_forecast(
    bench = bench,
    fit_model = make_linear_sv_fitter(
      compiled_model, chains, iter, warmup, seed, adapt_delta,
      max_treedepth, NULL, stationary_init, h1_scale
    ),
    forecast_one_step = forecast_linear_sv_one_step,
    update_state = update_linear_sv_state,
    refit_every = refit_every,
    model_name = "linearSV",
    save_fits = save_fits,
    seed = seed
  )
  result$posterior <- result$model$posterior
  result
}

if (isTRUE(getOption("benchmark.linear_sv_autorun", FALSE))) {
  bench <- load_benchmark_data()
  if (!exists("benchmark_results")) benchmark_results <- list()
  benchmark_results$linearSV <- run_linear_sv(bench = bench)
}
