library(rstan)

source("benchmark_utils.R")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

refit_every <- getOption("benchmark.refit_every", 13L)
save_rolling_fits <- getOption("benchmark.save_fits", FALSE)
stan_iter <- getOption("benchmark.stan_iter", 4000L)
stan_warmup <- getOption("benchmark.stan_warmup", 2000L)
stan_chains <- getOption("benchmark.stan_chains", 4L)

bench <- load_benchmark_data()

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

fit_ms_sv <- function(train_indices, previous_model, refit_id, bench) {
  stan_data <- ms_sv_stan_data(bench, train_indices)
  previous_post <- if (is.null(previous_model)) NULL else previous_model$posterior

  fit <- rstan::stan(
    file = "regime_switching_sv.stan",
    data = stan_data,
    chains = stan_chains,
    iter = stan_iter,
    warmup = stan_warmup,
    seed = 1666,
    init = function() init_ms_sv(stan_data, previous_post),
    control = list(adapt_delta = 0.90, max_treedepth = 10)
  )
  post <- rstan::extract(fit)
  n_draws <- length(post$p11)
  last_t <- length(train_indices)
  p_regime_1 <- post$filtered_prob[, last_t, 1]

  ms_diagnostic_pars <- c(
  "mu",
  "phi",
  "sigma_eta",
  "beta",
  "p11",
  "p22",
  "nu_minus2",
  "h")

  diagnostics = stan_fit_diagnostics(fit, pars = ms_diagnostic_pars)
  assert_stan_diagnostics(
    diagnostics = diagnostics,
    model_name = sprintf("Regime-switching SV refit %d", refit_id),
    strict = FALSE)

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

rolling_ms_sv <- rolling_forecast(
  bench = bench,
  fit_model = fit_ms_sv,
  forecast_one_step = forecast_ms_sv_one_step,
  update_state = update_ms_sv_state,
  refit_every = refit_every,
  model_name = "regimeSwitchingSV",
  save_fits = save_rolling_fits,
  seed = 1666
)
rolling_ms_sv$posterior <- rolling_ms_sv$model$posterior

if (!exists("benchmark_results")) {
  benchmark_results <- list()
}
benchmark_results$regimeSwitchingSV <- rolling_ms_sv
