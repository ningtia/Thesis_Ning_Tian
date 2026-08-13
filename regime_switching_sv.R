if (!exists("rolling_forecast", mode = "function", inherits = TRUE)) {
  source("benchmark_utils.R")
}

ms_sv_stan_data <- function(bench, train_indices) {
  list(
    T = length(train_indices),
    D = ncol(bench$X_all),
    y = bench$y_all[train_indices],
    X = bench$X_all[train_indices, , drop = FALSE]
  )
}

init_ms_sv <- function(stan_data, previous_post = NULL) {
  if (is.null(previous_post)) {
    mu <- sort(c(-1, 1) + log(mean(stan_data$y^2, na.rm = TRUE)))
    return(list(
      mu = mu,
      phi = c(0.90, 0.95),
      sigma_eta = c(0.25, 0.45),
      beta = rep(0, stan_data$D),
      h = rep(mean(mu), stan_data$T),
      p11 = 0.95,
      p22 = 0.95,
      nu_minus2 = 8
    ))
  }

  mu <- sort(apply(previous_post$mu, 2, stats::median))
  if (diff(mu) < 1e-4) mu[2] <- mu[1] + 0.1
  list(
    mu = mu,
    phi = pmin(apply(previous_post$phi, 2, stats::median), 0.995),
    sigma_eta = pmax(apply(previous_post$sigma_eta, 2, stats::median), 1e-4),
    beta = apply(previous_post$beta, 2, stats::median),
    h = rep(mean(mu), stan_data$T),
    p11 = min(stats::median(previous_post$p11), 0.995),
    p22 = min(stats::median(previous_post$p22), 0.995),
    nu_minus2 = max(stats::median(previous_post$nu_minus2), 1e-4)
  )
}

make_regime_switching_sv_fitter <- function(compiled_model,
                                            chains,
                                            iter,
                                            warmup,
                                            seed,
                                            adapt_delta,
                                            max_treedepth) {
  force(compiled_model)
  force(chains)
  force(iter)
  force(warmup)
  force(seed)
  force(adapt_delta)
  force(max_treedepth)

  function(train_indices, previous_model, refit_id, bench) {
    stan_data <- ms_sv_stan_data(bench, train_indices)
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

    ms_diagnostic_pars <- c(
      "mu", "phi", "sigma_eta", "beta", "p11", "p22", "nu_minus2", "h"
    )

    diagnostics <- stan_fit_diagnostics(fit, pars = ms_diagnostic_pars)
    assert_stan_diagnostics(
      diagnostics = diagnostics,
      model_name = sprintf("Regime-switching SV refit %d", refit_id),
      strict = FALSE
    )

    list(
      model = list(posterior = post),
      state = list(
        h = post$h[, last_t],
        regime = ifelse(stats::rbinom(n_draws, 1, p_regime_1) == 1L, 1L, 2L),
        draw_index = seq_len(n_draws)
      ),
      fit = fit,
      diagnostics = diagnostics
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
  parameter_index <- cbind(draw_index, next_regime)
  x_t <- bench$X_all[forecast_index, ]
  transition_mean <- post$mu[parameter_index] +
    post$phi[parameter_index] * (state$h - post$mu[parameter_index]) +
    as.vector(post$beta[draw_index, , drop = FALSE] %*% x_t)
  h_t <- transition_mean + post$sigma_eta[parameter_index] * stats::rnorm(n_draws)
  nu_t <- post$nu_minus2[draw_index] + 2
  scale_t <- sqrt(exp(h_t))

  list(
    variance_draws = exp(h_t) * nu_t / (nu_t - 2),
    return_draws = stats::rt(n_draws, df = nu_t) * scale_t,
    state_prior = list(
      h = h_t,
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
    h = forecast$state_prior$h[selected],
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
    adapt_delta = getOption("benchmark.stan_adapt_delta", 0.90),
    max_treedepth = getOption("benchmark.stan_max_treedepth", 10L)) {
  rstan::rstan_options(auto_write = TRUE)
  cores <- parallel::detectCores(logical = TRUE)
  if (is.na(cores)) cores <- 1L
  options(mc.cores = min(as.integer(chains), cores))
  compiled_model <- rstan::stan_model(file = "regime_switching_sv.stan")

  result <- rolling_forecast(
    bench = bench,
    fit_model = make_regime_switching_sv_fitter(
      compiled_model = compiled_model,
      chains = chains,
      iter = iter,
      warmup = warmup,
      seed = seed,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth
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
