# prior_predictive_check.R 
# The single most important check left: is the posterior for g_sd (the
# amplitude of the fitted nonlinearity) actually being driven by the data,
# or is it just the prior? 
# this script: the data block below is now the adopted production
# configuration (s_fixed=1.0, tau_w_scale=0.2), so the prior column of
# nnsv_prior_vs_posterior_all.csv is the authority, not this comment.
#
# The moderate/"main" design used throughout the recovery study has
# g_sd ~ 0.16 -- close to that prior BY CONSTRUCTION, since g_scale was not
# chosen to be far from what the prior implies. So "posterior looks like
# prior" there proves nothing either way, which is exactly the advisor's
# point 2. This script runs three scenarios and compares them:
#
#   zero    : g == 0 exactly (nnsv_simulation_null.csv). The model should
#             pull g_sd toward the prior's low edge, not report a spurious
#             nonlinearity -- the false-positive check.
#   main    : the design used throughout the recovery study (g_sd ~ 0.16,
#             close to the prior median). Kept only as a reference point.
#   strong  : g_sd ~ 0.4-0.5 with a leverage-driven ASYMMETRIC shape
#             (nnsv_simulation_strong.csv, built in simulation_study_nnsv.R),
#             deliberately outside the prior's 90% interval. If the
#             posterior still looks like the prior here, the nonlinear term
#             is not identified at this T, and that has to be reported as a
#             limitation, not papered over.
#
# None of the three "true" g paths are draws from the model's own NN prior:
# they come from an explicit sin(x) [+ leverage] function that has nothing
# to do with the W1/b1/w2/tau_w parameterisation the model estimates. So a
# posterior that tracks the true value is not just "reproducing how the data
# were drawn" -- that was the first thing to rule out (advisor's question).
#
# h_bar (sample level of the log-variance) is reported next to mu in every
# scenario: h_bar stays identified when phi -> 1 and mu does not.
#
# Run simulation_study_nnsv.R first if nnsv_simulation_strong.csv doesn't
# exist yet.

library(rstan)
library(posterior)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Reuse nn_sv_init()/nn_sv_linear_start() (chain init: data-driven mu/phi/
# sigma_eta, NN weights near zero) and benchmark_utils.R's
# stan_fit_diagnostics()/save_stan_diagnostic_plots() instead of
# reimplementing them here. benchmark.nn_autorun must be turned off FIRST --
# sourcing nonlinear_sv.R at its default option runs the full empirical
# rolling-forecast benchmark as a side effect (its bottom `if` block), which
# is not what loading it as a library should do. chdir = TRUE makes its own
# internal source("benchmark_utils.R") resolve against the repo root instead
# of this script's working directory (results/simulation/).
options(benchmark.nn_autorun = FALSE)
source("../../nonlinear_sv.R", chdir = TRUE)

model <- stan_model("nonlinear_sv.stan")

# DGP constants shared by all three scenarios (only g_scale/lev_scale differ
# between them; see simulation_study_nnsv.R). Needed to compute each
# scenario's true_value column below.
phi_true       <- 0.95
sigma_eta_true <- 0.25
nu_true        <- 10

run <- function(d, init_fn, ...) sampling(
  model, data = d, chains = 4, cores = 4,
  iter = 6000, warmup = 2000, refresh = 0, init = init_fn,
  control = list(adapt_delta = 0.995, max_treedepth = 12), ...
)

pull <- function(fit, par) as.numeric(rstan::extract(fit, par)[[par]])

# true_value = NA (e.g. tau_w, which has no DGP counterpart) skips bias/
# inside_95 rather than erroring.
compare <- function(fit_prior, fit_post, par, true_value = NA_real_) {
  a <- pull(fit_prior, par); b <- pull(fit_post, par)
  post_median <- median(b)
  post_sd   <- sd(b)
  post_q025 <- stats::quantile(b, .025, names = FALSE)
  post_q975 <- stats::quantile(b, .975, names = FALSE)
  data.frame(
    parameter    = par,
    true_value   = true_value,
    prior_median = median(a),
    prior_q05    = stats::quantile(a, .05, names = FALSE),
    prior_q95    = stats::quantile(a, .95, names = FALSE),
    post_median  = post_median,
    post_sd      = post_sd,
    post_q05     = stats::quantile(b, .05, names = FALSE),
    post_q95     = stats::quantile(b, .95, names = FALSE),
    post_q025    = post_q025,
    post_q975    = post_q975,
    bias         = if (is.na(true_value)) NA_real_ else post_median - true_value,
    inside_95    = if (is.na(true_value)) NA else (true_value >= post_q025 && true_value <= post_q975),
    # z-score: how many posterior sds the median sits from the truth
    z            = if (is.na(true_value)) NA_real_ else (post_median - true_value) / post_sd,
    # how much the data narrowed the parameter; ~1 means the data said nothing
    width_ratio  = (stats::quantile(b, .95, names = FALSE) - stats::quantile(b, .05, names = FALSE)) /
                   (stats::quantile(a, .95, names = FALSE) - stats::quantile(a, .05, names = FALSE)),
    row.names = NULL
  )
}

# ---------------------------------------------------------------------------
# Run one scenario end to end: prior + posterior fits (warm-started, NN
# weights near zero), the prior-vs-posterior table with true_value/bias/
# inside_95 (point 2), a saved density-overlay figure for g_sd and nu
# (point 4), the g-shape check against the true path (point 3), the h_bar
# readout, and per-chain HMC diagnostics + trace/rank/pairs plots for the
# posterior fit (point 6).
# ---------------------------------------------------------------------------
run_scenario <- function(label, sim_csv, seed_prior, seed_post, true_g_is_zero = FALSE) {
  message("\n=== scenario: ", label, " (", sim_csv, ") ===")
  sim_data <- read.csv(sim_csv)
  T_sim    <- nrow(sim_data)

  base_data <- list(
    T = T_sim, D = 1, K = 4,
    y = sim_data$y,
    X = matrix(sim_data$x[1:(T_sim - 1)], ncol = 1),
    use_student_t = 1,
    x_forecast = array(sim_data$x[T_sim], dim = 1),
    # 2026-08-26: synced to the ADOPTED production configuration so this check
    # describes the specification the thesis actually reports, not the interim
    # one. Every field below now matches nn_sv_default_config() in
    # nonlinear_sv.R plus run_models.R's NN_S_FIXED / NN_TAU_W_SCALE:
    #   s_fixed     0.50 -> 1.00   (NN_S_FIXED)
    #   tau_w_scale 0.50 -> 0.20   (NN_TAU_W_SCALE)
    #   phi_b       1.5  -> 2      (nn_phi_b)
    #   b1_scale    0.5  -> 1      (nn_b1_scale)
    # The previous values are archived with their outputs in
    # rep10-s0.5-3cases-3T/nnsv_prior_vs_posterior_all.csv.
    s_fixed = 1.00, tau_w_scale = 0.20,
    mu_scale = 5, phi_a = 20, phi_b = 2,
    sigma_eta_scale = 1, b1_scale = 1, nu_rate = 0.1,
    prior_only = 0,
    stationary_init = 0,   # fixed diffuse h[1]; removes the phi -> 1 funnel
    h1_scale = 2.0
  )
  prior_data <- base_data; prior_data$prior_only <- 1

  # True values for the recovery columns below. phi/sigma_eta/nu are DGP
  # constants; mu is level-shift corrected (mean(g)/(1-phi), same as
  # nnsv_main_true_parameters.csv) and g_sd/h_bar are realized quantities --
  # all three depend on this scenario's own simulated path, not a fixed
  # constant, except in the zero scenario where g == 0 makes the shift and
  # g_sd collapse to 0 automatically. tau_w has no DGP counterpart (NA).
  g_r <- sim_data$g_true[2:T_sim]
  true_vals <- c(
    mu        = 1.80 + mean(g_r) / (1 - phi_true),
    phi       = phi_true,
    sigma_eta = sigma_eta_true,
    nu        = nu_true,
    tau_w     = NA_real_,
    g_sd      = sd(g_r),
    h_bar     = mean(sim_data$h_true)
  )

  linear_start <- nn_sv_linear_start(base_data$y, base_data$X)
  init_fn <- function() nn_sv_init(base_data, previous_post = NULL, linear_start = linear_start)

  fit_prior <- run(prior_data, init_fn, seed = seed_prior)
  fit_post  <- run(base_data,  init_fn, seed = seed_post)

  # --- per-chain HMC diagnostics + trace/rank/pairs plot ---------
  # Only the posterior fit is checked this closely: prior_only drops the
  # likelihood, so there is nothing for those chains to get stuck on -- any
  # convergence issues there would be a geometry problem in the prior itself,
  # a much rarer failure mode than a data-driven funnel/ridge.
  sampler_params <- rstan::get_sampler_params(fit_post, inc_warmup = FALSE)
  max_td <- 12
  hmc_diagnostics <- data.frame(
    scenario          = label,
    chain             = seq_along(sampler_params),
    divergences       = sapply(sampler_params, function(p) sum(p[, "divergent__"])),
    max_treedepth_hit = sapply(sampler_params, function(p) sum(p[, "treedepth__"] >= max_td)),
    e_bfmi            = sapply(sampler_params, function(p) {
      e <- p[, "energy__"]; mean(diff(e)^2) / stats::var(e)
    }),
    row.names = NULL
  )
  write.csv(hmc_diagnostics, paste0("nnsv_prior_predictive_hmc_", label, ".csv"), row.names = FALSE)

  mcmc_diagnostics <- as.data.frame(posterior::summarise_draws(
    posterior::subset_draws(
      posterior::as_draws_array(fit_post),
      variable = c("mu", "phi", "sigma_eta", "nu", "tau_w", "g_sd", "h_bar", "lp__")
    )
  ))
  mcmc_diagnostics$scenario <- label
  write.csv(mcmc_diagnostics, paste0("nnsv_prior_predictive_mcmc_", label, ".csv"), row.names = FALSE)

  save_stan_diagnostic_plots(
    fit_post, paste0("nnsv_prior_predictive_trace_", label, ".pdf"),
    pars = c("mu", "phi", "sigma_eta", "tau_w", "g_sd", "h_bar"),
    pairs_pars = c("phi", "sigma_eta", "tau_w")
  )

  # --- prior vs posterior table, now with true_value/bias/inside_95
  tbl <- do.call(rbind, lapply(
    c("mu", "phi", "sigma_eta", "nu", "tau_w", "g_sd", "h_bar"),
    function(p) compare(fit_prior, fit_post, p, true_value = true_vals[[p]])))
  tbl$scenario <- label
  print(tbl, digits = 3)
  write.csv(tbl, paste0("nnsv_prior_vs_posterior_", label, ".csv"), row.names = FALSE)

  # --- prior/posterior density overlay for g_sd and nu -----------
  png(paste0("nnsv_prior_vs_posterior_", label, ".png"), width = 900, height = 450)
  op <- par(mfrow = c(1, 2))
  for (par_name in c("g_sd", "nu")) {
    a <- density(pull(fit_prior, par_name)); b <- density(pull(fit_post, par_name))
    plot(b, main = paste0(par_name, " (", label, ")"), xlab = par_name,
         xlim = range(a$x, b$x), ylim = c(0, max(a$y, b$y)), lwd = 2)
    lines(a, lty = 2)
    legend("topright", c("posterior", "prior"), lty = c(1, 2), lwd = c(2, 1), bty = "n")
  }
  par(op)
  dev.off()

  # --- shape of g, not just its size ------------------------------
  # g_sd only says the fitted wiggle is the right SIZE; this says whether it
  # is the right wiggle -- the advisor's suspicion is that sigma_eta comes
  # out too high because g has roughly the right size but the wrong shape,
  # pushing the leftover variance into sigma_eta.
  g_draws  <- rstan::extract(fit_post, "g")$g
  g_median <- apply(g_draws, 2, median)
  true_g_c <- sim_data$g_true[2:T_sim] - mean(sim_data$g_true[2:T_sim])

  if (true_g_is_zero) {
    shape <- data.frame(
      scenario = label, correlation = NA,
      rmse = sqrt(mean((g_median - true_g_c)^2)),
      sd_true_g = 0,
      note = "true g == 0; rmse alone measures spurious fitted signal"
    )
  } else {
    shape <- data.frame(
      scenario = label,
      correlation = cor(g_median, true_g_c),
      rmse = sqrt(mean((g_median - true_g_c)^2)),
      sd_true_g = sd(true_g_c),
      note = "rmse close to sd_true_g means no explanatory power at all"
    )
  }
  print(shape, digits = 3)

  # --- h_bar next to mu  -----------------------------------
  # True value and posterior are both already in `tbl`/its CSV now (point 2
  # fix); this is just a quick console readout, not the only record of it.
  h_bar_post <- pull(fit_post, "h_bar")
  message(label, " h_bar: true ", round(mean(sim_data$h_true), 3),
          " | posterior median ", round(median(h_bar_post), 3),
          " | 95% [", round(quantile(h_bar_post, .025), 3), ", ",
          round(quantile(h_bar_post, .975), 3), "]")

  list(tbl = tbl, shape = shape, hmc_diagnostics = hmc_diagnostics)
}

scenarios <- list(
  zero   = list(csv = "nnsv_simulation_null.csv",   zero = TRUE,  seed_prior = 11, seed_post = 12),
  main   = list(csv = "nnsv_simulation_main.csv",   zero = FALSE, seed_prior = 21, seed_post = 22),
  strong = list(csv = "nnsv_simulation_strong.csv", zero = FALSE, seed_prior = 31, seed_post = 32)
)

all_results <- lapply(names(scenarios), function(nm) {
  s <- scenarios[[nm]]
  run_scenario(nm, s$csv, s$seed_prior, s$seed_post, true_g_is_zero = s$zero)
})
names(all_results) <- names(scenarios)

combined_tbl   <- do.call(rbind, lapply(all_results, `[[`, "tbl"))
combined_shape <- do.call(rbind, lapply(all_results, `[[`, "shape"))
combined_hmc   <- do.call(rbind, lapply(all_results, `[[`, "hmc_diagnostics"))
write.csv(combined_tbl,   "nnsv_prior_vs_posterior_all.csv", row.names = FALSE)
write.csv(combined_shape, "nnsv_g_shape_all.csv",             row.names = FALSE)
write.csv(combined_hmc,   "nnsv_prior_predictive_hmc_all.csv", row.names = FALSE)
print(combined_tbl, digits = 3)
print(combined_shape, digits = 3)
print(combined_hmc, digits = 3)

cat("\nwidth_ratio near 1  -> the data are not informing that parameter.\n",
    "width_ratio near 0  -> the data dominate the prior.\n",
    "bias = post_median - true_value; inside_95 = true_value within the\n",
    "posterior's 95% interval (post_q025/post_q975) -- read these directly\n",
    "off the table now instead of eyeballing the density plot.\n",
    "Read g_sd across scenarios together, not in isolation:\n",
    "  zero:   post_median/post_q95 should sit near the prior's LOW edge\n",
    "          (correctly detecting nothing), not just a narrower interval.\n",
    "  strong: bias should be small and inside_95 TRUE, with width_ratio well\n",
    "          below the main scenario's.\n",
    "  If strong's posterior still tracks its prior, the nonlinear term is\n",
    "  not identified at this T -- report that, don't re-tune until it goes\n",
    "  away.\n",
    "Divergences/treedepth/E-BFMI per chain are in\n",
    "nnsv_prior_predictive_hmc_<scenario>.csv; rhat/ESS in\n",
    "nnsv_prior_predictive_mcmc_<scenario>.csv; trace/rank/pairs plots in\n",
    "nnsv_prior_predictive_trace_<scenario>.pdf.\n", sep = "")
