# Nonlinear neural-network stochastic-volatility model

library(rstan)
source("benchmark_utils.R")

configure_nn_sv_parallel <- function(chains) {
  cores <- parallel::detectCores(logical = TRUE)
  if (is.na(cores) || cores < 1L) cores <- 1L
  cores <- as.integer(cores)
  options(mc.cores = cores)
  list(available_cores = cores, concurrent_chains = min(as.integer(chains), cores))
}

nn_sv_default_config <- function() {
  warmup <- getOption("benchmark.nn_warmup", 2000L)
  sampling <- getOption("benchmark.nn_sampling", 4000L)
  list(
    stan_file = getOption("benchmark.nn_stan_file", "nonlinear_sv.stan"),
    chains = getOption("benchmark.nn_chains", 4L),
    warmup = warmup,
    iter = getOption("benchmark.nn_iter", warmup + sampling),
    adapt_delta = getOption("benchmark.nn_adapt_delta", 0.99),
    max_treedepth = getOption("benchmark.nn_max_treedepth", 12L),
    refit_every = getOption("benchmark.refit_every", 13),
    save_fits = getOption("benchmark.save_fits", FALSE),
    compute_loo = getOption("benchmark.nn_compute_loo", TRUE),
    s_fixed = getOption("benchmark.nn_s_fixed", 1),
    tau_w_scale = getOption("benchmark.nn_tau_w_scale", 0.2),
    mu_scale = getOption("benchmark.nn_mu_scale", 5),
    phi_a = getOption("benchmark.nn_phi_a", 20),
    phi_b = getOption("benchmark.nn_phi_b", 2),
    sigma_eta_scale = getOption("benchmark.nn_sigma_eta_scale", 1),
    b1_scale = getOption("benchmark.nn_b1_scale", 1),
    nu_rate = getOption("benchmark.nn_nu_rate", 0.1),
    stationary_init = getOption("benchmark.nn_stationary_init", 0L),
    h1_scale = getOption("benchmark.nn_h1_scale", 2.0),
    seed = getOption("benchmark.nn_seed", 6666L)
  )
}

validate_nn_sv_config <- function(config) {
  required <- c(
    "chains", "warmup", "iter", "adapt_delta", "max_treedepth",
    "s_fixed", "tau_w_scale", "mu_scale", "phi_a", "phi_b",
    "sigma_eta_scale", "b1_scale", "nu_rate", "stationary_init", "h1_scale"
  )
  missing <- setdiff(required, names(config))
  if (length(missing)) {
    stop("NN-SV configuration is missing: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  if (config$chains < 1L || config$warmup < 1L || config$iter <= config$warmup) {
    stop("NN-SV requires positive chains/warmup and iter > warmup.", call. = FALSE)
  }
  if (config$adapt_delta <= 0 || config$adapt_delta >= 1 ||
      config$max_treedepth < 1L) {
    stop("NN-SV requires 0 < adapt_delta < 1 and positive max_treedepth.",
         call. = FALSE)
  }
  positive <- c(
    s_fixed = config$s_fixed,
    tau_w_scale = config$tau_w_scale,
    mu_scale = config$mu_scale,
    phi_a = config$phi_a,
    phi_b = config$phi_b,
    sigma_eta_scale = config$sigma_eta_scale,
    b1_scale = config$b1_scale,
    nu_rate = config$nu_rate,
    h1_scale = config$h1_scale
  )
  if (any(!is.finite(positive)) || any(positive <= 0)) {
    stop("NN-SV scale and prior hyperparameters must be finite and positive.",
         call. = FALSE)
  }
  if (!isTRUE(config$stationary_init %in% c(0L, 1L, FALSE, TRUE))) {
    stop("NN-SV stationary_init must be 0/1 (or FALSE/TRUE).", call. = FALSE)
  }
  config
}

build_nn_sv_stan_data <- function(bench, train_indices, K,
                                  use_student_t, forecast_index = NULL,
                                  config = nn_sv_default_config()) {
  config <- validate_nn_sv_config(config)
  train_indices <- as.integer(train_indices)
  if (length(train_indices) < 2L) {
    stop("NN-SV needs at least two training observations.", call. = FALSE)
  }
  expected_indices <- seq.int(
    from = train_indices[1L],
    length.out = length(train_indices)
  )
  if (!identical(train_indices, expected_indices)) {
    stop("NN-SV train_indices must be strictly increasing and consecutive.",
         call. = FALSE)
  }
  if (is.null(forecast_index)) {
    forecast_index <- min(max(train_indices) + 1L, nrow(bench$all))
  }
  if (forecast_index < 1L || forecast_index > nrow(bench$all)) {
    stop("forecast_index is outside the benchmark data.", call. = FALSE)
  }

  X_lagged <- bench$X_all[train_indices[-1L], , drop = FALSE]
  storage.mode(X_lagged) <- "double"
  y <- as.numeric(bench$y_all[train_indices])
  nn_inputs <- cbind(
    y_positive = pmax(y[-length(y)], 0),
    y_negative = pmin(y[-length(y)], 0),
    X_lagged
  )
  z_center <- colMeans(nn_inputs)
  z_scale <- apply(nn_inputs, 2L, stats::sd)
  z_scale[!is.finite(z_scale) | z_scale <= 0] <- 1

  list(
    data = list(
      T = length(train_indices),
      D = ncol(X_lagged),
      K = as.integer(K),
      y = y,
      X = X_lagged,
      use_student_t = as.integer(isTRUE(use_student_t)),
      x_forecast = as.numeric(bench$X_all[forecast_index, ]),
      s_fixed = as.numeric(config$s_fixed),
      tau_w_scale = as.numeric(config$tau_w_scale),
      mu_scale = as.numeric(config$mu_scale),
      phi_a = as.numeric(config$phi_a),
      phi_b = as.numeric(config$phi_b),
      sigma_eta_scale = as.numeric(config$sigma_eta_scale),
      b1_scale = as.numeric(config$b1_scale),
      nu_rate = as.numeric(config$nu_rate),
      prior_only = 0L,
      stationary_init = as.integer(as.logical(config$stationary_init)),
      h1_scale = as.numeric(config$h1_scale)
    ),
    scaling = list(
      center = as.numeric(z_center),
      scale = as.numeric(z_scale)
    )
  )
}

nn_sv_conditional_volatility <- function(h, nu_minus2 = NULL) {
  if (is.null(nu_minus2) || length(nu_minus2) == 0L) {
    return(exp(h / 2))
  }
  nu <- as.numeric(nu_minus2) + 2
  exp(h / 2) * sqrt(nu / (nu - 2))
}

simulate_nn_sv_predictive <- function(post, y_previous, x_t,
                                      use_student_t = TRUE,
                                      h_previous = NULL,
                                      draw_index = NULL,
                                      s_fixed = 0.5,
                                      z_center = NULL,
                                      z_scale = NULL) {
  if (is.null(draw_index)) draw_index <- seq_len(length(post$mu))
  if (is.null(h_previous)) h_previous <- post$h[draw_index, ncol(post$h)]

  n_draws <- length(draw_index)
  z_raw <- c(max(y_previous, 0), min(y_previous, 0), as.numeric(x_t))
  if (is.null(z_center) || is.null(z_scale) ||
      length(z_center) != length(z_raw) || length(z_scale) != length(z_raw)) {
    stop("NN-SV prediction requires training-sample center/scale for D + 2 inputs.",
         call. = FALSE)
  }
  if (any(!is.finite(z_scale)) || any(z_scale <= 0)) {
    stop("NN-SV prediction scales must be finite and positive.", call. = FALSE)
  }
  z <- (z_raw - z_center) / z_scale
  nn_component <- vapply(seq_len(n_draws), function(i) {
    draw <- draw_index[i]
    hidden <- tanh(as.vector(post$W1[draw, , ] %*% z) + post$b1[draw, ])
    s_fixed * tanh(sum(post$w2[draw, ] * hidden)) - post$g_bar[draw]
  }, numeric(1))

  h_mean <- post$mu[draw_index] +
    post$phi[draw_index] * (h_previous - post$mu[draw_index]) + nn_component
  h_next <- h_mean + post$sigma_eta[draw_index] * stats::rnorm(n_draws)
  scale_next <- exp(h_next / 2)

  if (isTRUE(use_student_t)) {
    nu <- post$nu_minus2[draw_index] + 2
    return_draws <- stats::rt(n_draws, df = nu) * scale_next
    variance_draws <- exp(h_next) * nu / (nu - 2)
    volatility_draws <- nn_sv_conditional_volatility(h_next, post$nu_minus2[draw_index])
    log_density <- function(y) stats::dt(y / scale_next, df = nu, log = TRUE) - log(scale_next)
  } else {
    return_draws <- stats::rnorm(n_draws, sd = scale_next)
    variance_draws <- exp(h_next)
    volatility_draws <- scale_next
    log_density <- function(y) stats::dnorm(y, 0, scale_next, log = TRUE)
  }

  list(
    h_next = h_next,
    volatility_draws = volatility_draws,
    variance_draws = variance_draws,
    return_draws = return_draws,
    log_density_draws = log_density,
    state_prior = list(h = h_next, draw_index = draw_index)
  )
}

compute_nn_sv_loo <- function(log_lik) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    return(list(available = FALSE, result = NULL,
                message = "Package 'loo' is not installed; LOO-CV was skipped."))
  }
  result <- tryCatch(loo::loo(log_lik), error = function(e) e)
  if (inherits(result, "error")) {
    return(list(available = FALSE, result = NULL, message = conditionMessage(result)))
  }
  list(available = TRUE, result = result, message = NULL)
}

positive_ordered_initial_values <- function(x, minimum = 1e-4,
                                            separation = 1e-6) {
  values <- sort(pmax(as.numeric(x), minimum))
  if (length(values) > 1L) {
    for (i in 2:length(values)) {
      if (values[i] <= values[i - 1L]) {
        values[i] <- values[i - 1L] + separation
      }
    }
  }
  values
}

# Cheap linear-SV-style warm start for mu/phi/sigma_eta: regress the
# log-chi-square proxy z_t = log(y_t^2) + 1.2704 (the standard bias
# correction) on its own lag and the exogenous covariates. fit_nonlinear_sv()
# uses this to seed chains before the NN weights are even touched; extracted
# so other callers (e.g. the simulation-study diagnostics) can reuse it
# instead of running a separate full linear-SV Stan fit just for starting
# values.
nn_sv_linear_start <- function(y, X) {
  z <- log(pmax(y^2, .Machine$double.eps)) + 1.2704
  mu_start <- mean(z, na.rm = TRUE)
  phi_start <- 0.95
  sigma_start <- 0.25

  if (length(z) >= 10L) {
    proxy_data <- data.frame(z_now = z[-1L], z_lag = z[-length(z)], X)
    proxy_fit <- tryCatch(stats::lm(z_now ~ ., data = proxy_data), error = function(e) NULL)
    if (!is.null(proxy_fit)) {
      coefficients <- stats::coef(proxy_fit)
      if (is.finite(coefficients["z_lag"])) {
        phi_start <- min(max(unname(coefficients["z_lag"]), 0.50), 0.98)
      }
      residual_sd <- stats::sd(stats::residuals(proxy_fit), na.rm = TRUE)
      if (is.finite(residual_sd)) sigma_start <- min(max(residual_sd, 0.05), 1.00)
    }
  }
  list(
    mu = mu_start,
    phi = phi_start,
    phi_star = (phi_start + 1) / 2,
    sigma_eta = sigma_start
  )
}

nn_sv_init <- function(stan_data, previous_post = NULL, linear_start = NULL) {
  if (!is.null(previous_post)) {
    values <- list(
      mu = stats::median(previous_post$mu),
      phi_star = stats::median(previous_post$phi_star),
      sigma_eta = max(stats::median(previous_post$sigma_eta), 1e-4),
      W1 = apply(previous_post$W1, c(2, 3), stats::median),
      b1 = apply(previous_post$b1, 2, stats::median),
      w2_raw = positive_ordered_initial_values(
        apply(previous_post$w2_raw, 2, stats::median)
      ),
      tau_w = max(stats::median(previous_post$tau_w), 1e-4),
      eta_raw = rep(0, stan_data$T)
    )
    if (stan_data$use_student_t == 1L) {
      values$nu_minus2 <- array(max(stats::median(previous_post$nu_minus2), 1e-4), dim = 1L)
    }
    return(values)
  }

  d_nn <- stan_data$D + 2L
  values <- list(
    mu = linear_start$mu,
    phi_star = linear_start$phi_star,
    sigma_eta = linear_start$sigma_eta,
    W1 = matrix(stats::rnorm(stan_data$K * d_nn, 0, 0.1), stan_data$K, d_nn),
    b1 = stats::rnorm(stan_data$K, 0, 0.01),
    w2_raw = positive_ordered_initial_values(
      abs(stats::rnorm(stan_data$K, 0, 0.1))
    ),
    tau_w = 0.10,
    eta_raw = rep(0, stan_data$T)
  )
  if (stan_data$use_student_t == 1L) values$nu_minus2 <- array(8, dim = 1L)
  values
}

fit_nonlinear_sv <- function(bench, train_indices, K = 8L,
                             use_student_t = TRUE,
                             forecast_index = NULL,
                             previous_post = NULL,
                             config = nn_sv_default_config(),
                             seed = config$seed,
                             compute_loo = config$compute_loo,
                             make_diagnostic_plots = FALSE,
                             diagnostic_plot_file = "diagnostics_nonlinearSV.pdf",
                             strict_diagnostics = FALSE,
                             compiled_model = NULL) {
  config <- validate_nn_sv_config(config)
  train_indices <- as.integer(train_indices)
  if (is.null(forecast_index)) {
    forecast_index <- min(max(train_indices) + 1L, nrow(bench$all))
  }

  stan_bundle <- build_nn_sv_stan_data(
    bench = bench,
    train_indices = train_indices,
    K = K,
    use_student_t = use_student_t,
    forecast_index = forecast_index,
    config = config
  )
  stan_data <- stan_bundle$data
  nn_scaling <- stan_bundle$scaling

  linear_start <- NULL
  if (is.null(previous_post)) {
    linear_start <- nn_sv_linear_start(stan_data$y, stan_data$X)
  }

  if (is.null(compiled_model)) compiled_model <- rstan::stan_model(file = config$stan_file)
  fit <- rstan::sampling(
    object = compiled_model,
    data = stan_data,
    chains = config$chains,
    iter = config$iter,
    warmup = config$warmup,
    seed = seed,
    init = function() nn_sv_init(stan_data, previous_post, linear_start),
    control = list(adapt_delta = config$adapt_delta,
                   max_treedepth = config$max_treedepth)
  )
  post <- rstan::extract(fit)

  diagnostic_pars <- c("g_sd", "h_bar", "phi", "sigma_eta", "nu", "lp__")
  diagnostics <- stan_fit_diagnostics(fit, pars = diagnostic_pars)
  assert_stan_diagnostics(diagnostics, "Nonlinear SV", strict = strict_diagnostics)

  predictive <- simulate_nn_sv_predictive(
    post,
    y_previous = bench$y_all[tail(train_indices, 1L)],
    x_t = bench$X_all[forecast_index, ],
    use_student_t = use_student_t,
    s_fixed = config$s_fixed,
    z_center = nn_scaling$center,
    z_scale = nn_scaling$scale
  )
  y_draws <- predictive$return_draws
  q01 <- stats::quantile(y_draws, 0.01, names = FALSE)
  q05 <- stats::quantile(y_draws, 0.05, names = FALSE)
  predictive_summary <- list(
    variance = mean(predictive$variance_draws),
    volatility = mean(predictive$volatility_draws),
    VaR_01 = as.numeric(q01),
    ES_01 = mean(y_draws[y_draws <= q01]),
    VaR_05 = as.numeric(q05),
    ES_05 = mean(y_draws[y_draws <= q05])
  )

  parameter_draws <- list(
    mu = post$mu,
    phi = post$phi,
    sigma_eta = post$sigma_eta,
    tau_w = post$tau_w,
    g_sd = post$g_sd
  )

  if (!is.null(post$h_bar)) parameter_draws$h_bar <- post$h_bar
  if (!is.null(post$nu_minus2) && length(post$nu_minus2)) {
    parameter_draws$nu <- post$nu_minus2 + 2
  }
  parameter_summary <- data.frame(
    parameter = names(parameter_draws),
    mean = vapply(parameter_draws, mean, numeric(1)),
    median = vapply(parameter_draws, stats::median, numeric(1)),
    ci_05 = vapply(parameter_draws, stats::quantile, numeric(1), probs = 0.05, names = FALSE),
    ci_95 = vapply(parameter_draws, stats::quantile, numeric(1), probs = 0.95, names = FALSE),
    row.names = NULL
  )

  nu_minus2_draws <- if (isTRUE(use_student_t) && !is.null(post$nu_minus2) &&
                          length(post$nu_minus2)) as.numeric(post$nu_minus2) else NULL
  volatility_draws <- nn_sv_conditional_volatility(post$h, nu_minus2_draws)
  posterior_summary <- list(
    parameters = parameter_summary,
    filtered_volatility = data.frame(
      Date = as.Date(bench$all$Date[train_indices]),
      mean = colMeans(volatility_draws),
      median = apply(volatility_draws, 2, stats::median),
      p05 = apply(volatility_draws, 2, stats::quantile, probs = 0.05),
      p95 = apply(volatility_draws, 2, stats::quantile, probs = 0.95),
      stringsAsFactors = FALSE
    )
  )

  if (isTRUE(make_diagnostic_plots)) {
    save_stan_diagnostic_plots(
      fit,
      diagnostic_plot_file,
      pars = c("mu", "phi", "sigma_eta", "tau_w"),
      pairs_pars = c("phi", "sigma_eta", "tau_w")
    )
  }

  list(
    fit = fit,
    posterior = post,
    stan_data = stan_data,
    nn_scaling = nn_scaling,
    initialization = linear_start,
    diagnostics = diagnostics,
    posterior_summary = posterior_summary,
    predictive = predictive,
    predictive_summary = predictive_summary,
    loo = if (isTRUE(compute_loo)) compute_nn_sv_loo(post$log_lik) else NULL,
    K = as.integer(K),
    use_student_t = isTRUE(use_student_t),
    s_fixed = config$s_fixed,
    tau_w_scale = config$tau_w_scale,
    train_indices = train_indices,
    forecast_index = forecast_index
  )
}

fit_nn_sv_refit <- function(compiled_model, K, use_student_t, config,
                            train_indices, previous_model, refit_id, bench) {
  previous_post <- if (is.null(previous_model)) NULL else previous_model$posterior
  fitted <- fit_nonlinear_sv(
    bench,
    train_indices,
    K,
    use_student_t,
    previous_post = previous_post,
    config = config,
    compiled_model = compiled_model,
    seed = config$seed + 1000L * K + 100L * as.integer(use_student_t) + refit_id,
    compute_loo = FALSE
  )
  list(
    model = list(
      posterior = fitted$posterior,
      K = K,
      use_student_t = use_student_t,
      s_fixed = fitted$s_fixed,
      tau_w_scale = fitted$tau_w_scale,
      train_indices = train_indices,
      nn_scaling = fitted$nn_scaling,
      posterior_summary = fitted$posterior_summary,
      initialization = fitted$initialization
    ),
    state = list(
      h = fitted$posterior$h[, length(train_indices)],
      draw_index = seq_len(length(fitted$posterior$mu))
    ),
    fit = fitted$fit,
    diagnostics = fitted$diagnostics,
    usable = isTRUE(fitted$diagnostics$all_checks_passed)
  )
}

run_nn_sv_preflight <- function(compiled_model, bench, candidate, config) {
  train_indices <- bench$split_rows$train
  fitted <- fit_nn_sv_refit(
    compiled_model,
    candidate$K,
    candidate$use_student_t,
    config,
    train_indices,
    NULL,
    1L,
    bench
  )
  passed <- isTRUE(fitted$usable)
  list(
    passed = passed,
    fitted = fitted,
    diagnostics = fitted$diagnostics,
    train_indices = train_indices,
    candidate = candidate,
    message = if (passed) "Initial NN-SV Stan diagnostic check (training window only) passed."
              else "Initial NN-SV Stan diagnostic check (training window only) failed; validation/test were not run."
  )
}

default_nn_candidates <- function() {
  list(
    list(K = 4L, use_student_t = TRUE,  label = "K4_Student_t"),
    list(K = 4L, use_student_t = FALSE, label = "K4_Gaussian"),
    list(K = 8L, use_student_t = TRUE,  label = "K8_Student_t"),
    list(K = 8L, use_student_t = FALSE, label = "K8_Gaussian")
  )
}

make_nn_sv_fitter <- function(compiled_model, candidate, candidate_config, first_refit = NULL) {
  function(train_indices, previous_model, refit_id, bench) {
    if (identical(as.integer(refit_id), 1L) && !is.null(first_refit)) return(first_refit)
    fit_nn_sv_refit(
      compiled_model,
      candidate$K,
      candidate$use_student_t,
      candidate_config,
      train_indices,
      previous_model,
      refit_id,
      bench
    )
  }
}

forecast_nn_sv_one_step <- function(model, state, forecast_index, bench) {
  simulate_nn_sv_predictive(
    model$posterior,
    bench$y_all[forecast_index - 1L],
    bench$X_all[forecast_index, ],
    model$use_student_t,
    state$h,
    state$draw_index,
    model$s_fixed,
    model$nn_scaling$center,
    model$nn_scaling$scale
  )
}

update_nn_sv_state <- function(model, state, forecast, observed_y, observation_index, bench) {
  log_weights <- forecast$log_density_draws(observed_y)
  weights <- exp(log_weights - max(log_weights))
  selected <- sample.int(length(weights), length(weights), replace = TRUE, prob = weights)
  list(h = forecast$state_prior$h[selected],
       draw_index = forecast$state_prior$draw_index[selected])
}

run_nonlinear_sv <- function(bench = load_benchmark_data(),
                             candidates = default_nn_candidates(),
                             architecture_metric = getOption("benchmark.architecture_metric", "qlike"),
                             config = nn_sv_default_config()) {
  parallel_config <- configure_nn_sv_parallel(config$chains)
  compiled_model <- rstan::stan_model(file = config$stan_file)

  runner <- function(candidate, forecast_indices, refit_every) {
    candidate_config <- config
    candidate_config$refit_every <- refit_every
    first_refit <- NULL

    rolling_forecast(
      bench,
      make_nn_sv_fitter(compiled_model, candidate, candidate_config, first_refit),
      forecast_nn_sv_one_step,
      update_nn_sv_state,
      forecast_indices,
      candidate_config$refit_every,
      paste0("NNSV_", candidate$label),
      candidate_config$save_fits,
      candidate_config$seed + candidate$K + as.integer(candidate$use_student_t)
    )
  }

  result <- select_rolling_architecture(
    bench,
    candidates,
    runner,
    metric = architecture_metric,
    refit_every = config$refit_every
  )
  result$posterior <- result$model$posterior
  result$posterior_summary <- result$model$posterior_summary
  result$initialization <- result$model$initialization
  result$loo <- if (isTRUE(config$compute_loo)) {
    compute_nn_sv_loo(result$model$posterior$log_lik)
  } else NULL
  result$sampler_config <- config
  result$parallel <- parallel_config
  result
}

if (isTRUE(getOption("benchmark.nn_autorun", TRUE))) {
  bench <- load_benchmark_data()
  rolling_nn_sv <- run_nonlinear_sv(bench)
  if (!exists("benchmark_results")) benchmark_results <- list()
  benchmark_results$nonlinearSV <- rolling_nn_sv
}
