# Marginal effects, partial dependence, input ablation and IRFs for the fitted
# NN-SV model. Everything here is computed from the STORED POSTERIOR DRAWS --
# no re-estimation -- because g_t is a static function of z_t and h_t is an
# AR(1) in g_t, so both the response surface and its dynamic propagation are
# available in closed form given the draws.
if (dir.exists("Rlib")) .libPaths(c(normalizePath("Rlib"), .libPaths()))
out <- "results/tables"
options(benchmark.nn_autorun = FALSE)
source("benchmark_utils.R")

bench <- load_benchmark_data()
nn <- readRDS("results/models/nonlinearSV.rds")
post <- nn$model$posterior
K    <- nn$model$K
s    <- nn$model$s_fixed
ctr  <- nn$model$nn_scaling$center
scl  <- nn$model$nn_scaling$scale
tri  <- nn$model$train_indices
cat("K =", K, " s_fixed =", s, " |train| =", length(tri), "\n")

input_names <- c("y_positive", "y_negative", bench$covariates)
D_nn <- length(input_names)
stopifnot(length(ctr) == D_nn)

# --- rebuild the network input matrix Z exactly as build_nn_sv_stan_data() did
y  <- bench$y_all[tri]
Xl <- bench$X_all[tri[-1L], , drop = FALSE]
raw <- cbind(y_positive = pmax(y[-length(y)], 0),
             y_negative = pmin(y[-length(y)], 0), Xl)
Z <- sweep(sweep(raw, 2L, ctr, "-"), 2L, scl, "/")
N <- nrow(Z)
cat("Z:", N, "x", ncol(Z), "\n")

# --- thin the draws: 2000 is ample for these summaries and keeps it fast
n_draws <- length(post$tau_w)
idx <- round(seq(1, n_draws, length.out = min(2000L, n_draws)))
cat("using", length(idx), "of", n_draws, "draws\n")

W1 <- post$W1[idx, , , drop = FALSE]   # [draw, K, D_nn]
b1 <- post$b1[idx, , drop = FALSE]     # [draw, K]
w2 <- post$w2[idx, , drop = FALSE]     # [draw, K]
phi <- post$phi[idx]

g_of <- function(Zmat, d) {            # uncentred network output for draw d
  A <- tanh(Zmat %*% t(W1[d, , ]) + matrix(b1[d, ], nrow(Zmat), K, byrow = TRUE))
  s * tanh(as.vector(A %*% w2[d, ]))
}

# =====================================================================
# 1. INPUT ABLATION -- how much of the amplitude does each input carry?
#    g_sd recomputed with input j held at its training mean (z_j = 0).
# =====================================================================
cat("\n--- input ablation ---\n"); flush.console()
abl <- matrix(NA_real_, length(idx), D_nn, dimnames = list(NULL, input_names))
kep <- matrix(NA_real_, length(idx), D_nn, dimnames = list(NULL, input_names))
full <- numeric(length(idx))
for (d in seq_along(idx)) {
  gf <- g_of(Z, d)
  full[d] <- sd(gf - mean(gf))
  for (j in seq_len(D_nn)) {
    Zj <- Z; Zj[, j] <- 0
    gj <- g_of(Zj, d)
    abl[d, j] <- sd(gj - mean(gj))
    Zk <- matrix(0, N, D_nn); Zk[, j] <- Z[, j]   # input j alone
    gk <- g_of(Zk, d)
    kep[d, j] <- sd(gk - mean(gk))
  }
}
ablation <- data.frame(
  input = input_names,
  g_sd_full = median(full),
  g_sd_ablated = apply(abl, 2L, median),
  drop = median(full) - apply(abl, 2L, median),
  drop_pct = 100 * (median(full) - apply(abl, 2L, median)) / median(full),
  drop_q025 = apply(full - abl, 2L, quantile, .025),
  drop_q975 = apply(full - abl, 2L, quantile, .975),
  # amplitude left when ONLY this input varies -- the yardstick against which
  # the joint "retained" figures below must be read, since even an input the
  # network ignores leaves a non-zero residual through posterior weight noise
  retained = apply(kep, 2L, median),
  row.names = NULL
)
ablation <- ablation[order(-ablation$drop), ]
print(ablation, digits = 3)
write.csv(ablation, file.path(out, "nnsv_input_ablation.csv"), row.names = FALSE)

# =====================================================================
# 1b. JOINT ABLATION -- the single-input ablation above is marginal, and the
#     single drops sum to far less than the amplitude, so most of the fitted
#     function lives in interactions. The motivating hypothesis is itself a
#     JOINT condition (high CPU *and* falling EUA *and* a negative return), so
#     it can only be tested by moving those inputs together. Two statistics:
#       drop      -- amplitude lost when the whole group is held at its mean
#                    (group removed, everything else free)
#       retained  -- amplitude left when only the group varies and all other
#                    inputs are held at their means (group kept, alone). This
#                    is the decisive one: it preserves every interaction
#                    *within* the group, so a near-zero value says the group
#                    cannot generate the nonlinearity even jointly.
#     Superadditivity, drop minus the sum of the group's single-input drops,
#     measures how much of the group's effect is interactive.
# =====================================================================
cat("\n--- joint ablation ---\n"); flush.console()
groups <- list(
  hypothesis_triple = c("GCPU_baseline", "c_t", "y_negative"),
  cpu_and_carbon    = c("GCPU_baseline", "c_t"),
  all_climate       = c("GCPU_baseline", "c_t", "l_t"),
  financial         = c("itraxx", "VSTOXX")
)
stopifnot(all(unlist(groups) %in% input_names))

jdrop <- jkeep <- matrix(NA_real_, length(idx), length(groups),
                         dimnames = list(NULL, names(groups)))
for (d in seq_along(idx)) {
  for (gi in seq_along(groups)) {
    cols <- match(groups[[gi]], input_names)
    Zd <- Z; Zd[, cols] <- 0                       # group removed
    gd <- g_of(Zd, d); jdrop[d, gi] <- sd(gd - mean(gd))
    Zk <- matrix(0, N, D_nn); Zk[, cols] <- Z[, cols]  # group alone
    gk <- g_of(Zk, d); jkeep[d, gi] <- sd(gk - mean(gk))
  }
}

single_sum <- vapply(groups, function(g)
  rowSums(full - abl[, match(g, input_names), drop = FALSE]), numeric(length(idx)))
drop_draws <- full - jdrop            # amplitude lost by removing the group
inter_draws <- drop_draws - single_sum  # interactive part of that loss

q <- function(M, p) apply(M, 2L, quantile, p, names = FALSE)
joint <- data.frame(
  group        = names(groups),
  inputs       = vapply(groups, paste, character(1), collapse = "+"),
  g_sd_full    = median(full),
  drop         = apply(drop_draws, 2L, median),
  drop_pct     = 100 * apply(drop_draws, 2L, median) / median(full),
  drop_q025    = q(drop_draws, .025),
  drop_q975    = q(drop_draws, .975),
  sum_singles  = apply(single_sum, 2L, median),
  interaction  = apply(inter_draws, 2L, median),
  inter_q025   = q(inter_draws, .025),
  inter_q975   = q(inter_draws, .975),
  retained     = apply(jkeep, 2L, median),
  retained_pct = 100 * apply(jkeep, 2L, median) / median(full),
  retained_q025 = q(jkeep, .025),
  retained_q975 = q(jkeep, .975),
  row.names = NULL
)
print(joint, digits = 3)
write.csv(joint, file.path(out, "nnsv_joint_ablation.csv"), row.names = FALSE)

cat("\nReading: 'drop' = amplitude lost when the group is held at its mean;",
    "\n'retained' = amplitude left when ONLY the group varies.",
    "\nAn interval on 'drop' or 'retained' that contains zero (or a retained",
    "\nvalue far below the 0.06 detection floor) says the group does not carry",
    "\nthe nonlinearity, jointly or interactively.\n")

# =====================================================================
# 2. PARTIAL DEPENDENCE -- g as a function of one input, others at mean.
# =====================================================================
cat("\n--- partial dependence ---\n"); flush.console()
grid_z <- seq(-2.5, 2.5, length.out = 41)
pd_rows <- list()
for (j in seq_len(D_nn)) {
  Zg <- matrix(0, length(grid_z), D_nn); Zg[, j] <- grid_z
  M <- vapply(seq_along(idx), function(d) g_of(Zg, d), numeric(length(grid_z)))
  M <- M - matrix(colMeans(M), nrow(M), ncol(M), byrow = TRUE)  # centre per draw
  pd_rows[[j]] <- data.frame(
    input = input_names[j], z = grid_z,
    raw_value = ctr[j] + grid_z * scl[j],
    median = apply(M, 1L, median),
    q05 = apply(M, 1L, quantile, .05),
    q95 = apply(M, 1L, quantile, .95),
    row.names = NULL)
}
pd <- do.call(rbind, pd_rows)
write.csv(pd, file.path(out, "nnsv_partial_dependence.csv"), row.names = FALSE)

cpu <- pd[pd$input == "GCPU_baseline", ]
cat("\nPartial dependence on climate-policy uncertainty:\n")
print(cpu[cpu$z %in% grid_z[seq(1, 41, by = 5)],
          c("z", "raw_value", "median", "q05", "q95")], digits = 3)

# curvature check: slope in the lower vs upper half of the range
lo <- cpu[cpu$z <= 0, ]; hi <- cpu[cpu$z >= 0, ]
slope <- function(d) coef(lm(median ~ z, data = d))[2]
cat("\nslope of g wrt CPU, lower half (z<=0):", round(slope(lo), 4),
    "| upper half (z>=0):", round(slope(hi), 4), "\n")

# =====================================================================
# 3. IMPULSE RESPONSE -- a +1 sd shock to input j at t=0, effect on E[h_{t+k}].
#    g_t is static in z_t, so the impulse enters h once and decays at phi^k.
# =====================================================================
cat("\n--- impulse responses ---\n"); flush.console()
horizons <- 0:12
irf_rows <- list()
for (j in seq_len(D_nn)) {
  z0 <- matrix(0, 1, D_nn); z1 <- z0; z1[1, j] <- 1
  dg <- vapply(seq_along(idx), function(d) g_of(z1, d) - g_of(z0, d), numeric(1))
  for (k in horizons) {
    v <- dg * phi^k
    irf_rows[[length(irf_rows) + 1L]] <- data.frame(
      input = input_names[j], horizon = k,
      median = median(v), q05 = quantile(v, .05, names = FALSE),
      q95 = quantile(v, .95, names = FALSE), row.names = NULL)
  }
}
irf <- do.call(rbind, irf_rows)
write.csv(irf, file.path(out, "nnsv_impulse_responses.csv"), row.names = FALSE)
cat("\nIRF of log-variance to a +1 sd shock, horizon 0:\n")
h0 <- irf[irf$horizon == 0, ]
print(h0[order(-abs(h0$median)), ], digits = 3, row.names = FALSE)

cat("\nCumulative response over 12 weeks (median), top inputs:\n")
cum <- aggregate(median ~ input, data = irf, FUN = sum)
print(cum[order(-abs(cum$median)), ], digits = 3, row.names = FALSE)

cat("\n==== done ====\n")
