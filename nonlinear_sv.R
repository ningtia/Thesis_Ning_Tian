################################################################################
## Nonlinear neural-network stochastic volatility model (NN-SV)
##
## Implements Sections 8.7--8.8:
##   * standardized x_{t-1} supplied by load_benchmark_data();
##   * log-chi-square linear-SV proxy initialization;
##   * 4 chains, 1,500 warmup + 2,000 sampling iterations by default;
##   * posterior, convergence, predictive-density, VaR/ES, and LOO outputs;
##   * expanding-window one-step-ahead forecast evaluation through
##     rolling_forecast().
################################################################################

library(rstan)

source("benchmark_utils.R")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

nn_sv_default_config <- function() {
  warmup <- getOption("benchmark.nn_warmup", 1500L)
  sampling <- getOption("benchmark.nn_sampling", 2000L)

  list(
    stan_file = getOption("benchmark.nn_stan_file", "nonlinear_sv.stan"),
    chains = getOption("benchmark.nn_chains", 4L),
    warmup = warmup,
    iter = getOption("benchmark.nn_iter", warmup + sampling),
    max_treedepth = getOption("benchmark.nn_max_treedepth", 12L),
    # Use 0.95 only after observed divergences, as specified in Section 8.7.
    adapt_delta = getOption("benchmark.nn_adapt_delta", 0.90),
    refit_every = getOption("benchmark.refit_every", 13L),
    save_fits = getOption("benchmark.save_fits", FALSE),
    compute_loo = getOption("benchmark.nn_compute_loo", TRUE),
    seed = getOption("benchmark.nn_seed", 6666L)
  )
}

validate_nn_sv_config <- function(config) {
  required <- c("chains", "warmup", "iter", "max_treedepth", "adapt_delta")
  if (!all(required %in% names(config))) {
    stop("NN-SV sampler configuration is incomplete.")
  }
  if (config$chains < 1L || config$warmup < 1L || config$iter <= config$warmup) {
    stop("NN-SV requires positive chains/warmup and iter > warmup.")
  }
  if (config$max_treedepth < 10L || config$adapt_delta <= 0 || config$adapt_delta >= 1) {
    stop("NN-SV requires max_treedepth >= 10 and 0 < adapt_delta < 1.")
  }
  invisible(config)
}

# X_all is standardized with the initial (pre-2022) training sample in
# benchmark_utils.R.  Crucially, the same scaling is then retained for all
# future expanding-window fits, avoiding any look-ahead standardization.
build_nn_sv_stan_data <- function(bench,
                                  train_indices,
                                  K,
                                  use_student_t,
                                  forecast_index = NULL) {
  train_indices <- as.integer(train_indices)
  if (length(train_indices) < 2L) {
    stop("NN-SV needs at least two training observations.")
  }
  if (is.null(forecast_index)) {
    forecast_index <- min(max(train_indices) + 1L, nrow(bench$all))
  }
  if (forecast_index < 1L || forecast_index > nrow(bench$all)) {
    stop("forecast_index is outside the benchmark data.")
  }

  # In nonlinear_sv.stan, X[i, ] drives h_{i+1}.  The processed covariate in
  # benchmark row t is already x_{t-1}; hence global rows 2:T correctly align
  # with y_2,...,y_T without using y_t or future information.
  X_lagged <- bench$X_all[train_indices[-1L], , drop = FALSE]
  storage.mode(X_lagged) <- "double"

  list(
    T = length(train_indices),
    D = ncol(X_lagged),
    K = as.integer(K),
    y = as.numeric(bench$y_all[train_indices]),
    X = X_lagged,
    use_student_t = as.integer(isTRUE(use_student_t)),
    x_forecast = as.numeric(bench$X_all[forecast_index, ])
  )
}

# A fast log-chi-square linear SV pre-estimate.  For Gaussian observations,
# E[log(y_t^2) | h_t] is h_t - 1.2704.  This provides stable mu, phi, and
# sigma_eta start values without adding an expensive second MCMC run before
# every quarterly NN-SV re-estimation.
preestimate_linear_sv <- function(y, X_lagged) {
  y <- as.numeric(y)
  z <- log(pmax(y^2, .Machine$double.eps)) + 1.2704
  T_len <- length(z)

  mu_start <- mean(z, na.rm = TRUE)
  phi_start <- 0.95
  sigma_start <- 0.25
  beta_start <- stats::setNames(rep(0, ncol(X_lagged)), colnames(X_lagged))

  if (T_len >= 10L) {
    regression_data <- data.frame(
      z_now = z[-1L],
      z_lag = z[-T_len],
      X_lagged
    )
    proxy_fit <- tryCatch(
      stats::lm(z_now ~ ., data = regression_data),
      error = function(e) NULL
    )
    if (!is.null(proxy_fit)) {
      coefficients <- stats::coef(proxy_fit)
      if (is.finite(coefficients["z_lag"])) {
        phi_start <- min(max(unname(coefficients["z_lag"]), 0.50), 0.98)
      }
      common_beta <- intersect(names(beta_start), names(coefficients))
      beta_start[common_beta] <- coefficients[common_beta]
      residual_sd <- stats::sd(stats::residuals(proxy_fit), na.rm = TRUE)
      if (is.finite(residual_sd)) {
        sigma_start <- min(max(residual_sd, 0.05), 1.00)
      }
    }
  }

  list(
    mu = mu_start,
    phi = phi_start,
    phi_raw = atanh(phi_start),
    sigma_eta = sigma_start,
    beta = unname(beta_start)
  )
}

nn_sv_init <- function(stan_data,
                       previous_post = NULL,
                       linear_start = NULL) {
  d_nn <- stan_data$D + 3L

  if (!is.null(previous_post)) {
     init_values <-list(
      mu = stats::median(previous_post$mu),
      phi_raw = stats::median(previous_post$phi_raw),
      sigma_eta = max(stats::median(previous_post$sigma_eta), 1e-4),
      W1 = apply(previous_post$W1, c(2, 3), stats::median),
      b1 = apply(previous_post$b1, 2, stats::median),
      w2 = apply(previous_post$w2, 2, stats::median),
      b2 = stats::median(previous_post$b2),
      s = max(stats::median(previous_post$s), 1e-4),
      tau_w = max(stats::median(previous_post$tau_w), 1e-4),
      eta_raw = rep(0, stan_data$T)
    )
    if (stan_data$use_student_t == 1L) {
      init_values$nu_minus2 <- max(
        stats::median(previous_post$nu_minus2),
        1e-4
      )
    }
    return(init_values)
  }

  if (is.null(linear_start)) {
    linear_start <- preestimate_linear_sv(stan_data$y, stan_data$X)
  }
  init_values <- list(
    mu = linear_start$mu,
    phi_raw = linear_start$phi_raw,
    sigma_eta = linear_start$sigma_eta,
    W1 = matrix(stats::rnorm(stan_data$K * d_nn, 0, 0.01), stan_data$K, d_nn),
    b1 = stats::rnorm(stan_data$K, 0, 0.01),
    w2 = stats::rnorm(stan_data$K, 0, 0.01),
    b2 = 0,
    s = 0.10,
    tau_w = 0.10,
    eta_raw = rep(0, stan_data$T)
  )

   if (stan_data$use_student_t == 1L) {
    init_values$nu_minus2 <- 8
  }

  init_values
}


assert_nn_sv_diagnostics <- function(diagnostics, strict = FALSE) {
  problems <- c(
    if (!isTRUE(diagnostics$rhat_pass)) "R-hat >= 1.01",
    if (!isTRUE(diagnostics$bulk_ess_pass)) "bulk-ESS <= 400",
    if (!is.na(diagnostics$min_tail_ess) && !isTRUE(diagnostics$tail_ess_pass)) "tail-ESS <= 400",
    if (diagnostics$divergence_sum > 0) "divergent transitions",
    if (!isTRUE(diagnostics$ebfmi_pass)) "E-BFMI <= 0.30"
  )
  if (length(problems) > 0L) {
    message("NN-SV diagnostic warning: ", paste(problems, collapse = "; "), ".")
    if (isTRUE(strict)) {
      stop("NN-SV convergence checks failed.")
    }
  }
  invisible(diagnostics)
}

summarize_draws <- function(x, parameter) {
  x <- as.numeric(x)
  data.frame(
    parameter = parameter,
    mean = mean(x),
    median = stats::median(x),
    ci_05 = as.numeric(stats::quantile(x, 0.05, names = FALSE)),
    ci_95 = as.numeric(stats::quantile(x, 0.95, names = FALSE)),
    stringsAsFactors = FALSE
  )
}

summarize_nn_sv_posterior <- function(post, dates) {

  parameter_parts <- list(
    summarize_draws(post$mu, "mu"),
    summarize_draws(tanh(post$phi_raw), "phi"),
    summarize_draws(post$sigma_eta, "sigma_eta"),
    summarize_draws(post$tau_w, "tau_w"),
    summarize_draws(post$s, "s"),
    summarize_draws(post$b2, "b2")
  )

  if (!is.null(post$nu_minus2) && length(post$nu_minus2) > 0L) {
    parameter_parts <- append(parameter_parts,
      list(summarize_draws(
          post$nu_minus2 + 2, "nu")))
  }

  parameter_summary <- do.call(
    rbind, parameter_parts
  )


  volatility_draws <- exp(post$h / 2)
  
  filtered_volatility <- data.frame(
    Date = as.Date(dates),
    mean = colMeans(volatility_draws),
    median = apply(volatility_draws, 2, stats::median),
    p05 = apply(volatility_draws, 2, stats::quantile, probs = 0.05),
    p95 = apply(volatility_draws, 2, stats::quantile, probs = 0.95),
    stringsAsFactors = FALSE
  )

  list(
    parameters = parameter_summary,
    filtered_volatility = filtered_volatility
  )
}

nn_transition <- function(post, draw_index, h_previous, y_previous, x_t) {

  n_draws <- length(draw_index)
  y_positive <- max(y_previous, 0)
  y_negative <- min(y_previous, 0)

  nn_component <- vapply(seq_len(n_draws), function(i) {

    draw <- draw_index[i]
    z <- c(h_previous[i], y_positive, y_negative, x_t)
    hidden <- tanh(as.vector(post$W1[draw, , ] %*% z) + post$b1[draw, ])
    post$s[draw] * tanh(sum(post$w2[draw, ] * hidden) + post$b2[draw])
  }, numeric(1))

  post$mu[draw_index] +
    tanh(post$phi_raw[draw_index]) * (h_previous - post$mu[draw_index]) + 
    nn_component
}

simulate_nn_sv_predictive <- function(post,
                                      y_previous,
                                      x_t,
                                      use_student_t = TRUE,
                                      h_previous = NULL,
                                      draw_index = NULL) {
  if (is.null(draw_index)) draw_index <- seq_len(length(post$mu))
  if (is.null(h_previous)) h_previous <- post$h[draw_index, ncol(post$h)]
  n_draws <- length(draw_index)
  h_mean <- nn_transition(post, draw_index, h_previous, y_previous, x_t)
  h_next <- h_mean + post$sigma_eta[draw_index] * stats::rnorm(n_draws)
  scale_next <- sqrt(exp(h_next))

  if (isTRUE(use_student_t)) {
    nu <- post$nu_minus2[draw_index] + 2
    y_next <- stats::rt(n_draws, df = nu) * scale_next
    variance_draws <- exp(h_next) * nu / (nu - 2)
    log_density <- function(y) {
      stats::dt(y / scale_next, df = nu, log = TRUE) - log(scale_next)
    }
  } else {
    y_next <- stats::rnorm(n_draws, mean = 0, sd = scale_next)
    variance_draws <- exp(h_next)
    log_density <- function(y) {
      stats::dnorm(y, mean = 0, sd = scale_next, log = TRUE)
    }
  }

  list(
    h_next = h_next,
    volatility_draws = sqrt(exp(h_next)),
    variance_draws = variance_draws,
    return_draws = y_next,
    log_density_draws = log_density,
    state_prior = list(h = h_next, draw_index = draw_index)
  )
}

summarize_predictive_distribution <- function(predictive) {
  y <- predictive$return_draws
  q01 <- stats::quantile(y, 0.01, names = FALSE)
  q05 <- stats::quantile(y, 0.05, names = FALSE)
  list(
    variance = mean(predictive$variance_draws),
    volatility = mean(predictive$volatility_draws),
    VaR_01 = as.numeric(q01),
    ES_01 = mean(y[y <= q01]),
    VaR_05 = as.numeric(q05),
    ES_05 = mean(y[y <= q05])
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

save_nn_sv_diagnostic_plots <- function(fit,
                                        file = "diagnostics_nonlinearSV.pdf",
                                        pars = c("mu", "phi_raw", "sigma_eta", "tau_w", "s")) {
  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    warning("Package 'bayesplot' is not installed; diagnostic plots were skipped.")
    return(invisible(FALSE))
  }
  available <- intersect(pars, fit@model_pars)
  if (length(available) == 0L) return(invisible(FALSE))

  draws <- as.matrix(fit, pars = available)
  grDevices::pdf(file, width = 10, height = 7)
  on.exit(grDevices::dev.off(), add = TRUE)
  print(bayesplot::mcmc_trace(draws, pars = available))
  print(bayesplot::mcmc_rank_hist(draws, pars = available))
  if (length(available) >= 2L) {
    print(bayesplot::mcmc_pairs(draws, pars = available[1:2]))
  }
  invisible(TRUE)
}

# Single complete NN-SV estimation.  This function is useful for the
# validation architecture runs and also for an in-sample estimation appendix.
fit_nonlinear_sv <- function(bench,
                             train_indices,
                             K = 8L,
                             use_student_t = TRUE,
                             forecast_index = NULL,
                             previous_post = NULL,
                             config = nn_sv_default_config(),
                             seed = config$seed,
                             compute_loo = config$compute_loo,
                             make_diagnostic_plots = FALSE,
                             diagnostic_plot_file = "diagnostics_nonlinearSV.pdf",
                             strict_diagnostics = FALSE) {
  validate_nn_sv_config(config)
  stan_data <- build_nn_sv_stan_data(
    bench = bench,
    train_indices = train_indices,
    K = K,
    use_student_t = use_student_t,
    forecast_index = forecast_index
  )
  linear_start <- if (is.null(previous_post)) {
    preestimate_linear_sv(stan_data$y, stan_data$X)
  } else {
    NULL
  }

  fit <- rstan::stan(
    file = config$stan_file,
    data = stan_data,
    chains = config$chains,
    iter = config$iter,
    warmup = config$warmup,
    seed = seed,
    init = function() nn_sv_init(stan_data, previous_post, linear_start),
    control = list(
      adapt_delta = config$adapt_delta,
      max_treedepth = config$max_treedepth
    )
  )
  post <- rstan::extract(fit)
  diagnostics <- stan_fit_diagnostics(fit)
  assert_nn_sv_diagnostics(diagnostics, strict = strict_diagnostics)

  forecast_index <- if (is.null(forecast_index)) {
    min(max(train_indices) + 1L, nrow(bench$all))
  } else {
    forecast_index
  }
  predictive <- simulate_nn_sv_predictive(
    post = post,
    y_previous = bench$y_all[max(train_indices)],
    x_t = bench$X_all[forecast_index, ],
    use_student_t = use_student_t
  )

  if (isTRUE(make_diagnostic_plots)) {
    save_nn_sv_diagnostic_plots(fit, diagnostic_plot_file)
  }

  list(
    fit = fit,
    posterior = post,
    stan_data = stan_data,
    initialization = linear_start,
    diagnostics = diagnostics,
    posterior_summary = summarize_nn_sv_posterior(
      post,
      bench$all$Date[train_indices]
    ),
    predictive = predictive,
    predictive_summary = summarize_predictive_distribution(predictive),
    loo = if (isTRUE(compute_loo)) compute_nn_sv_loo(post$log_lik) else NULL,
    K = as.integer(K),
    use_student_t = isTRUE(use_student_t),
    train_indices = train_indices,
    forecast_index = forecast_index
  )
}

make_nn_sv_rolling_fitter <- function(K, use_student_t, config) {
  force(K)
  force(use_student_t)
  force(config)

  function(train_indices, previous_model, refit_id, bench) {
    previous_post <- if (is.null(previous_model)) NULL else previous_model$posterior
    fitted <- fit_nonlinear_sv(
      bench = bench,
      train_indices = train_indices,
      K = K,
      use_student_t = use_student_t,
      previous_post = previous_post,
      config = config,
      seed = config$seed + 1000L * K + 100L * as.integer(use_student_t) + refit_id,
      # LOO is retained for the final selected fit, not every rolling refit.
      compute_loo = FALSE
    )
    n_draws <- length(fitted$posterior$mu)

    list(
      model = list(
        posterior = fitted$posterior,
        K = K,
        use_student_t = use_student_t,
        train_indices = train_indices,
        posterior_summary = fitted$posterior_summary,
        initialization = fitted$initialization
      ),
      state = list(
        h = fitted$posterior$h[, length(train_indices)],
        draw_index = seq_len(n_draws)
      ),
      fit = fitted$fit,
      diagnostics = fitted$diagnostics
    )
  }
}

forecast_nn_sv_one_step <- function(model, state, forecast_index, bench) {
  predictive <- simulate_nn_sv_predictive(
    post = model$posterior,
    y_previous = bench$y_all[forecast_index - 1L],
    x_t = bench$X_all[forecast_index, ],
    use_student_t = model$use_student_t,
    h_previous = state$h,
    draw_index = state$draw_index
  )
  predictive
}

update_nn_sv_state <- function(model, state, forecast, observed_y,
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

default_nn_candidates <- function() {
  list(
    list(K = 8L, use_student_t = FALSE, label = "K8_Gaussian"),
    list(K = 8L, use_student_t = TRUE, label = "K8_Student_t"),
    list(K = 16L, use_student_t = FALSE, label = "K16_Gaussian"),
    list(K = 16L, use_student_t = TRUE, label = "K16_Student_t")
  )
}

run_nn_candidate <- function(bench, candidate, forecast_indices, config) {
  rolling_forecast(
    bench = bench,
    fit_model = make_nn_sv_rolling_fitter(
      K = candidate$K,
      use_student_t = candidate$use_student_t,
      config = config
    ),
    forecast_one_step = forecast_nn_sv_one_step,
    update_state = update_nn_sv_state,
    forecast_indices = forecast_indices,
    refit_every = config$refit_every,
    model_name = paste0("NNSV_", candidate$label),
    save_fits = config$save_fits,
    seed = config$seed + candidate$K + as.integer(candidate$use_student_t)
  )
}

last_refit_loo <- function(result, bench, config) {
  model <- result$model
  if (!isTRUE(config$compute_loo)) return(NULL)
  compute_nn_sv_loo(model$posterior$log_lik)
}

run_nonlinear_sv <- function(bench = load_benchmark_data(),
                             candidates = default_nn_candidates(),
                             architecture_metric = getOption(
                               "benchmark.architecture_metric", "qlike"
                             ),
                             config = nn_sv_default_config()) {
  validate_nn_sv_config(config)

  runner <- function(candidate, forecast_indices, refit_every) {
    candidate_config <- config
    candidate_config$refit_every <- refit_every
    run_nn_candidate(bench, candidate, forecast_indices, candidate_config)
  }
  result <- select_rolling_architecture(
    bench = bench,
    candidates = candidates,
    run_candidate = runner,
    metric = architecture_metric,
    refit_every = config$refit_every
  )
  result$posterior <- result$model$posterior
  result$posterior_summary <- result$model$posterior_summary
  result$initialization <- result$model$initialization
  result$loo <- last_refit_loo(result, bench, config)
  result$sampler_config <- config
  result
}

# Calling source("nonlinear_sv.R") runs the complete 8.7--8.8 protocol and
# stores the result alongside the other benchmark models.  Set
# options(benchmark.nn_autorun = FALSE) before source() when importing only
# the helper functions, or call run_nonlinear_sv() with custom settings.
if (isTRUE(getOption("benchmark.nn_autorun", TRUE))) {
  bench <- load_benchmark_data()
  rolling_nn_sv <- run_nonlinear_sv(bench = bench)
  if (!exists("benchmark_results")) {
    benchmark_results <- list()
  }
  benchmark_results$nonlinearSV <- rolling_nn_sv
}
