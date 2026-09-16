# =====================================================================
# 03_conditional_economics.R
#
# Re-analysis of the stored forecasts, no re-estimation:
#   1. Diebold-Mariano tests computed INSIDE the high- and low-CPU
#      sub-samples, for all three loss functions.
#   2. The volatility-targeting table with a buy-and-hold row added.
#   3. The series behind two figures: wealth paths, and rolling realised
#      volatility of each strategy against the 15% target.
#
# Each model's .rds is loaded, its (small) forecast_table extracted, and the
# object dropped before the next one -- three of them carry full posterior
# draws and run to hundreds of megabytes.
# =====================================================================

if (dir.exists("Rlib")) .libPaths(c(normalizePath("Rlib"), .libPaths()))
options(benchmark.evaluate_autorun = FALSE, benchmark.garch_autorun = FALSE,
        benchmark.har_autorun = FALSE, benchmark.linear_sv_autorun = FALSE,
        benchmark.ms_sv_autorun = FALSE, benchmark.nn_autorun = FALSE)
source("benchmark_utils.R")
source("evaluate_results.R")

TABLES <- "results/tables"
ECON   <- "results/economic"
dir.create(ECON, showWarnings = FALSE, recursive = TRUE)

MODELS <- c(sGARCH = "sGARCH", gjrGARCH = "gjrGARCH", eGARCH = "eGARCH",
            harRV = "harRV", linearSV = "linearSV",
            regimeSwitchingSV = "regimeSwitchingSV", nonlinearSV = "nonlinearSV")

bench <- load_benchmark_data()
regimes <- cpu_regime(bench, "GCPU_baseline")[, c("forecast_date", "CPU_regime")]
regimes$forecast_date <- as.Date(regimes$forecast_date)

cat("loading forecast tables\n"); flush.console()
fc <- list()
for (m in MODELS) {
  path <- file.path("results/models", paste0(m, ".rds"))
  if (!file.exists(path)) { warning("missing ", path); next }
  obj <- readRDS(path)
  tab <- obj$forecast_table
  tab <- tab[tab$split == "test", , drop = FALSE]
  tab$forecast_date <- as.Date(tab$forecast_date)
  fc[[m]] <- tab[order(tab$forecast_date), ]
  rm(obj); gc(verbose = FALSE)
  cat("  ", m, ": ", nrow(fc[[m]]), " test forecasts\n", sep = "")
}

dates <- Reduce(intersect, lapply(fc, function(x) as.character(x$forecast_date)))
dates <- sort(as.Date(dates))
cat("common test dates:", length(dates), "\n")

align <- function(tab) tab[match(dates, tab$forecast_date), , drop = FALSE]
fc <- lapply(fc, align)
actual <- fc[[1]]$actual_return
regime <- regimes$CPU_regime[match(dates, regimes$forecast_date)]
cat("regime counts:", paste(names(table(regime)), table(regime), sep = "=", collapse = " "), "\n")

# =====================================================================
# 1. DM tests within each CPU regime
# =====================================================================
loss_series <- function(tab, which_loss) {
  switch(which_loss,
    qlike        = qlike_loss(tab$actual_return, tab$variance),
    negative_lps = -tab$log_score,
    fz05         = fz0_score(tab$actual_return, tab$VaR_05, tab$ES_05, 0.05))
}

rows <- list()
for (lname in c("qlike", "negative_lps", "fz05")) {
  L <- sapply(fc, loss_series, which_loss = lname)
  for (r in c("high", "low")) {
    keep <- regime == r & stats::complete.cases(L)
    Lr <- L[keep, , drop = FALSE]
    nm <- colnames(Lr)
    for (i in seq_along(nm)) for (j in seq_along(nm)) {
      if (i >= j) next
      tst <- dm_test_pair(Lr[, i], Lr[, j], nm[i], nm[j])
      rows[[length(rows) + 1L]] <- data.frame(
        loss = lname, regime = r, n = nrow(Lr),
        model_A = nm[i], model_B = nm[j],
        mean_diff = mean(Lr[, i] - Lr[, j]),
        statistic = tst$statistic, p_value = tst$p_value,
        better = ifelse(mean(Lr[, i] - Lr[, j]) < 0, nm[i], nm[j]),
        row.names = NULL)
    }
  }
}
dm_regime <- do.call(rbind, rows)
write.csv(dm_regime, file.path(TABLES, "dm_by_cpu_regime.csv"), row.names = FALSE)

cat("\n--- DM tests within HIGH-CPU weeks, QLIKE, pairs involving a state-dependent model ---\n")
sub <- dm_regime[dm_regime$loss == "qlike" & dm_regime$regime == "high" &
                 (dm_regime$model_A %in% c("nonlinearSV", "regimeSwitchingSV") |
                  dm_regime$model_B %in% c("nonlinearSV", "regimeSwitchingSV")), ]
print(sub[order(sub$p_value), c("model_A", "model_B", "mean_diff", "statistic", "p_value", "better")],
      digits = 3, row.names = FALSE)
cat("\nsignificant at 5% anywhere in the regime-conditional DM matrix: ",
    sum(dm_regime$p_value < 0.05, na.rm = TRUE), " of ", nrow(dm_regime), "\n", sep = "")

# =====================================================================
# 2. Portfolio table with buy-and-hold
# =====================================================================
TARGET <- 0.15; LEVCAP <- 2; GAMMA <- 3; PPY <- 52

metrics_from_weights <- function(w, r_pct) {
  pr <- w * r_pct / 100
  wealth <- cumprod(1 + pr)
  dd <- 1 - wealth / cummax(wealth)
  ar <- mean(pr) * PPY
  av <- sd(pr) * sqrt(PPY)
  list(wealth = wealth, ret = pr, annual_return = ar, annual_vol = av,
       sharpe = if (av > 0) ar / av else NA_real_,
       max_drawdown = max(dd), cer = ar - 0.5 * GAMMA * av^2)
}

target_pct <- 100 * TARGET / sqrt(PPY)
port <- list()
for (m in names(fc)) {
  w <- pmin(target_pct / sqrt(pmax(fc[[m]]$variance, 1e-12)), LEVCAP)
  port[[m]] <- metrics_from_weights(w, actual)
}
port[["buyAndHold"]] <- metrics_from_weights(rep(1, length(actual)), actual)

port_tab <- do.call(rbind, lapply(names(port), function(m) data.frame(
  Model = m,
  annual_return = port[[m]]$annual_return,
  annual_vol = port[[m]]$annual_vol,
  dev_from_target_pp = 100 * (port[[m]]$annual_vol - TARGET),
  sharpe = port[[m]]$sharpe,
  max_drawdown = port[[m]]$max_drawdown,
  CER = port[[m]]$cer,
  row.names = NULL)))
port_tab <- port_tab[order(abs(port_tab$dev_from_target_pp)), ]
write.csv(port_tab, file.path(TABLES, "portfolio_with_buyhold.csv"), row.names = FALSE)
cat("\n--- volatility targeting, buy-and-hold included ---\n")
print(port_tab, digits = 4, row.names = FALSE)

# =====================================================================
# 3. Figure series
# =====================================================================
wealth_df <- data.frame(date = dates)
for (m in names(port)) wealth_df[[m]] <- port[[m]]$wealth
write.csv(wealth_df, file.path(ECON, "wealth_paths.csv"), row.names = FALSE)

WIN <- 13L
roll_vol <- function(x) {
  out <- rep(NA_real_, length(x))
  for (i in seq_along(x)) if (i >= WIN) out[i] <- sd(x[(i - WIN + 1L):i]) * sqrt(PPY)
  out
}
rv_df <- data.frame(date = dates)
for (m in names(port)) rv_df[[m]] <- roll_vol(port[[m]]$ret)
write.csv(rv_df, file.path(ECON, "rolling_strategy_volatility.csv"), row.names = FALSE)

cat("\nmean 13-week rolling realised vol (annualised), target 0.15:\n")
print(round(sapply(rv_df[-1], mean, na.rm = TRUE), 4))
cat("\nshare of weeks with rolling vol above target:\n")
print(round(sapply(rv_df[-1], function(v) mean(v > TARGET, na.rm = TRUE)), 3))

cat("\n==== done ====\n")
