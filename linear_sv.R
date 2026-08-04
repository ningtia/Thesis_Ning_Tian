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

linear_sv_stan_data <- function(bench, train_indices) {
  y_train <- bench$y_all[train_indices]
  list(
    T = length(train_indices),
    D = ncol(bench$X_all),
    y = y_train,
    X = bench$X_all[train_indices, , drop = FALSE]
  )
}

linear_sv_init <- function(stan_data, previous_post = NULL) {
  if (is.null(previous_post)) {
    return(list(
      mu = log(mean(actual_return_variance(stan_data$y), na.rm = TRUE)),
      phi_raw = atanh(0.95),
      sigma_eta = 0.25,
      beta = rep(0, stan_data$D),
      eta_raw = rep(0, stan_data$T),
      nu_minus2 = 8
    ))
  }

  list(
    mu = stats::median(previous_post$mu),
    phi_raw = stats::median(previous_post$phi_raw),
    sigma_eta = max(stats::median(previous_post$sigma_eta), 1e-4),
    beta = apply(previous_post$beta, 2, stats::median),
    eta_raw = rep(0, stan_data$T),
    nu_minus2 = max(stats::median(previous_post$nu_minus2), 1e-4)
  )
}

fit_linear_sv <- function(train_indices, previous_model, refit_id, bench) {
  stan_data <- linear_sv_stan_data(bench, train_indices)
  previous_post <- if (is.null(previous_model)) NULL else previous_model$posterior

  fit <- rstan::stan(
    file = "linear_sv.stan",
    data = stan_data,
    chains = stan_chains,
    iter = stan_iter,
    warmup = stan_warmup,
    seed = 666,
    init = function() linear_sv_init(stan_data, previous_post),
    control = list(adapt_delta = 0.90, max_treedepth = 10)
  )
  post <- rstan::extract(fit)
  n_draws <- length(post$mu)

  list(
    model = list(posterior = post),
    state = list(
      h = post$h[, length(train_indices)],
      draw_index = seq_len(n_draws)
    ),
    fit = fit,
    diagnostics = stan_fit_diagnostics(fit)
  )
}

forecast_linear_sv_one_step <- function(model, state, forecast_index, bench) {
  post <- model$posterior
  draw_index <- state$draw_index
  n_draws <- length(draw_index)
  x_t <- bench$X_all[forecast_index, ]

  transition_mean <- post$mu[draw_index] +
    tanh(post$phi_raw[draw_index]) * (state$h - post$mu[draw_index]) +
    as.vector(post$beta[draw_index, , drop = FALSE] %*% x_t)
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

rolling_linear_sv <- rolling_forecast(
  bench = bench,
  fit_model = fit_linear_sv,
  forecast_one_step = forecast_linear_sv_one_step,
  update_state = update_linear_sv_state,
  refit_every = refit_every,
  model_name = "linearSV",
  save_fits = save_rolling_fits,
  seed = 666
)
rolling_linear_sv$posterior <- rolling_linear_sv$model$posterior

if (!exists("benchmark_results")) {
  benchmark_results <- list()
}
benchmark_results$linearSV <- rolling_linear_sv
