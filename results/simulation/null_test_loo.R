# null_test_loo.R
#
# "Is there a nonlinearity?" answered by model comparison rather than by asking
# NUTS to sample a degenerate posterior.
#
# Under the null DGP the true NN amplitude is zero, tau_w -> 0, and W1 and b1
# become unidentified. That funnel sharpens as T grows, which is why the null
# scenario gives a 12% usable rate at T = 312 and 0/10 at T = 1000. Comparing
# the NN model against the linear SV by LOO never visits that region.
#
# Expected pattern:
#   zero DGP   -> linear model preferred, or elpd difference within ~2 se of 0
#   main DGP   -> NN model preferred by more than 2 se
#   strong DGP -> NN model preferred, but see the s_fixed ceiling issue

library(rstan)
# library(loo)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ---------------------------------------------------------------------------
# Shared settings. Both models take the same data list.
# ---------------------------------------------------------------------------
make_data <- function(d, K = 4, s_fixed = 1, tau_w_scale = 0.2) {
  T_sim <- nrow(d)
  list(
    T = T_sim, D = 1, K = K,
    y = d$y,
    X = matrix(d$x[1:(T_sim - 1)], ncol = 1),
    use_student_t = 1,
    x_forecast = array(d$x[T_sim], dim = 1),
    s_fixed = s_fixed, tau_w_scale = tau_w_scale,
    mu_scale = 5,
    phi_a = 20, phi_b = 2,        # keeps phi off the unit root
    sigma_eta_scale = 1, b1_scale = 0.5, nu_rate = 0.1,
    prior_only = 0,
    stationary_init = 0,            # fixed diffuse h[1]; removes the phi -> 1 funnel
    h1_scale = 2.0
  )
}

fit_one <- function(model, data, seed) sampling(
  model, data = data, chains = 4, cores = 4,
  iter = 4000, warmup = 2000, refresh = 0, init_r = 0.5,
  control = list(adapt_delta = 0.99, max_treedepth = 12), seed = seed
)

m_nn  <- stan_model("nonlinear_sv_v3.stan")
m_lin <- stan_model("linear_sv.stan")

compare_one <- function(d, label, seed = 1) {
  sd_ <- make_data(d)
  f_nn  <- fit_one(m_nn,  sd_, seed)
  f_lin <- fit_one(m_lin, sd_, seed + 1)

  ll_nn  <- loo::extract_log_lik(f_nn,  "log_lik", merge_chains = FALSE)
  ll_lin <- loo::extract_log_lik(f_lin, "log_lik", merge_chains = FALSE)
  l_nn  <- loo::loo(ll_nn,  r_eff = loo::relative_eff(exp(ll_nn)))
  l_lin <- loo::loo(ll_lin, r_eff = loo::relative_eff(exp(ll_lin)))

  cmp <- loo::loo_compare(list(linear = l_lin, nn = l_nn))
  print(cmp)

  # elpd difference in favour of the NN model, with its standard error
  diff <- cmp["nn", "elpd_diff"] - cmp["linear", "elpd_diff"]
  se   <- max(cmp[, "se_diff"])

  # divergences, for the record -- but they no longer gate the conclusion
  ndiv <- function(f) sum(sapply(get_sampler_params(f, inc_warmup = FALSE),
                                 function(p) sum(p[, "divergent__"])))
  data.frame(
    scenario   = label,
    elpd_diff_nn_minus_linear = diff,
    se_diff    = se,
    ratio      = diff / se,
    verdict    = ifelse(abs(diff) < 2 * se, "no evidence of nonlinearity",
                        ifelse(diff > 0, "nonlinearity supported",
                                         "linear preferred")),
    div_nn     = ndiv(f_nn),
    div_linear = ndiv(f_lin),
    # the amplitude, alongside the false-positive floor from the null DGP
    g_sd_median = median(as.numeric(rstan::extract(f_nn, "g_sd")$g_sd)),
    row.names = NULL
  )
}

# ---------------------------------------------------------------------------
# Run over the three scenarios. Expects the CSVs written by
# simulation_study_nnsv.R (nnsv_simulation_leverage.csv from the older design
# doesn't exist any more -- (c) was renamed "strong", see that script's header).
# ---------------------------------------------------------------------------
res <- rbind(
  compare_one(read.csv("nnsv_simulation_null.csv"),     "zero",   seed = 11),
  compare_one(read.csv("nnsv_simulation_main.csv"),     "main",   seed = 21),
  compare_one(read.csv("nnsv_simulation_strong.csv"),   "strong", seed = 31)
)
print(res, digits = 3)
write.csv(res, "nnsv_loo_null_test.csv", row.names = FALSE)

cat("\nFalse-positive floor from your own null scenario: g_sd ~ 0.055 at T = 312.\n",
    "An estimated amplitude below that should not be called a nonlinearity.\n", sep = "")
