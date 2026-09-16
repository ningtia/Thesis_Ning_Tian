# economic_results_pipeline.R

# ---- 1. Volatility-targeting portfolio -------------------------------------
# Scale exposure so realized volatility matches a target,
# using the model's variance forecast each period.

# 2026-08-26 bug fix: actual_return (and therefore forecast_variance, which
# is fit to actual_return^2) is on a PERCENT scale -- e.g. 3.09 means a 3.09%
# weekly move, so sqrt(forecast_variance) ~= 3 means a 3-percentage-point
# weekly vol forecast. target_vol above was left as a raw decimal fraction
# (0.15/sqrt(52) ~= 0.0208), so weight = target_vol / sqrt(forecast_variance)
# was comparing a decimal to a percent-point figure -- off by exactly 100x.
# The resulting weight was ~0.007 instead of ~0.7, max_leverage (2) never
# bound, and the "15%-vol-targeting" portfolio actually ran at ~0.17% annual
# vol: essentially flat, so every model's Sharpe/CER just reflected noise
# around the sample's slightly negative mean return rather than any actual
# volatility-timing skill. Multiplying target_vol by 100 puts it on the same
# percent scale as sqrt(forecast_variance) before the division; the /100
# on portfolio_return below (converting percent returns to a decimal wealth
# multiplier) is unchanged and still correct.
portfolio_metrics <- function(actual_return, forecast_variance,
                              target_annual_vol = 0.15,
                              max_leverage = 2,
                              risk_aversion = 3,
                              periods_per_year = 52) {
  target_vol <- 100 * target_annual_vol / sqrt(periods_per_year)
  weight <- pmin(target_vol / sqrt(pmax(forecast_variance, 1e-12)), max_leverage)

  portfolio_return <- weight * actual_return / 100
  wealth <- cumprod(1 + portfolio_return)
  drawdown <- 1 - wealth / cummax(wealth)

  annual_return <- mean(portfolio_return) * periods_per_year
  annual_vol <- sd(portfolio_return) * sqrt(periods_per_year)

  list(
    wealth = wealth,
    portfolio_return = portfolio_return,
    sharpe = if (annual_vol > 0) annual_return / annual_vol else NA_real_,
    annual_return = annual_return,
    annual_vol = annual_vol,
    max_drawdown = max(drawdown),
    cer = annual_return - 0.5 * risk_aversion * annual_vol^2
  )
}

# ---- 2. VaR-implied capital cost -------------------------------------------
# Capital a risk manager would hold against the forecast VaR, and how often
# a loss actually broke through it

capital_cost <- function(actual_return, VaR, alpha) {
  list(
    alpha = alpha,
    capital_pct = mean(pmax(-VaR, 0), na.rm = TRUE),
    breach_rate = mean(actual_return < VaR, na.rm = TRUE)
  )
}

# ---- 3. One summary row per model ------------------------------------------

model_economic_summary <- function(model_name, result,
                                   target_annual_vol = 0.15,
                                   max_leverage = 2,
                                   risk_aversion = 3) {
  forecast <- result$forecast_table
  if (is.null(forecast)) return(NULL)
  forecast <- forecast[forecast$split == "test", ]
  if (!nrow(forecast)) return(NULL)
  forecast <- forecast[order(forecast$forecast_date), ]

  p <- portfolio_metrics(
    forecast$actual_return, forecast$variance,
    target_annual_vol, max_leverage, risk_aversion
  )
  c01 <- capital_cost(forecast$actual_return, forecast$VaR_01, 0.01)
  c05 <- capital_cost(forecast$actual_return, forecast$VaR_05, 0.05)

  row <- data.frame(
    Model = model_name,
    N = nrow(forecast),
    Sharpe = p$sharpe,
    Annual_Return = p$annual_return,
    Annual_Volatility = p$annual_vol,
    Max_Drawdown = p$max_drawdown,
    CER = p$cer,
    Capital_01pct = c01$capital_pct,
    Breach_Rate_01 = c01$breach_rate,
    Capital_05pct = c05$capital_pct,
    Breach_Rate_05 = c05$breach_rate
  )
  list(row = row, wealth = p$wealth, portfolio_return = p$portfolio_return,
       dates = forecast$forecast_date)
}

# ---- 3b. Buy-and-hold benchmark row -----------------------------------------
# 2026-08-27: added at the advisor's request ("add the buy-and-hold row to
# the economic table"). Full, unlevered exposure every week (weight = 1,
# no variance forecast involved) on the same test-period actual returns
# every model is scored on, so Sharpe/CER/drawdown are directly comparable
# to the seven vol-targeting rows: this answers "does timing the exposure
# with a variance forecast beat just holding the asset?"
#
# The 1%/5% VaR-implied capital columns need *some* VaR series even though
# buy-and-hold has no variance forecast to supply one; the standard choice
# for a "naive" VaR benchmark is the unconditional historical quantile,
# estimated once on the training split (never on validation/test) and held
# fixed across the whole test period -- a static historical-simulation VaR,
# not a forecast that adapts week to week. That asymmetry (constant vs.
# time-varying VaR) is inherent to what "buy-and-hold" means here and should
# be flagged wherever this row is discussed, not smoothed over.
buy_and_hold_summary <- function(result,
                                 target_annual_vol = 0.15,
                                 risk_aversion = 3,
                                 bench = NULL) {
  forecast <- result$forecast_table
  forecast <- forecast[forecast$split == "test", ]
  forecast <- forecast[order(forecast$forecast_date), ]

  actual_return <- forecast$actual_return
  portfolio_return <- actual_return / 100
  wealth <- cumprod(1 + portfolio_return)
  drawdown <- 1 - wealth / cummax(wealth)

  annual_return <- mean(portfolio_return) * 52
  annual_vol <- sd(portfolio_return) * sqrt(52)
  sharpe <- if (annual_vol > 0) annual_return / annual_vol else NA_real_
  cer <- annual_return - 0.5 * risk_aversion * annual_vol^2

  train_return <- if (!is.null(bench)) bench$y_all[bench$split_rows$train] else NULL
  var01 <- if (!is.null(train_return)) stats::quantile(train_return, 0.01, names = FALSE) else NA_real_
  var05 <- if (!is.null(train_return)) stats::quantile(train_return, 0.05, names = FALSE) else NA_real_
  c01 <- capital_cost(actual_return, rep(var01, length(actual_return)), 0.01)
  c05 <- capital_cost(actual_return, rep(var05, length(actual_return)), 0.05)

  row <- data.frame(
    Model = "BuyAndHold",
    N = nrow(forecast),
    Sharpe = sharpe,
    Annual_Return = annual_return,
    Annual_Volatility = annual_vol,
    Max_Drawdown = max(drawdown),
    CER = cer,
    Capital_01pct = c01$capital_pct,
    Breach_Rate_01 = c01$breach_rate,
    Capital_05pct = c05$capital_pct,
    Breach_Rate_05 = c05$breach_rate
  )
  list(row = row, wealth = wealth, portfolio_return = portfolio_return,
       dates = forecast$forecast_date)
}

# ---- 4. Wealth-path plot ----------------------------------------------------
# 2026-08-27 rewrite at the advisor's request: real calendar dates on the
# x-axis (was a bare 1..N step index); nonlinear SV drawn thick black so it
# reads as the model the thesis is about, every other fitted model thin grey
# so the comparison set doesn't visually compete with it, and buy-and-hold
# added as a dashed black reference line (distinguishable from both the
# thick-solid nn-SV line and the thin-solid grey cluster).

plot_wealth_paths <- function(summaries, file) {
  pdf(file, width = 9, height = 6)
  on.exit(dev.off())

  model_names <- vapply(summaries, function(x) x$row$Model, character(1))
  wealth <- lapply(summaries, `[[`, "wealth")
  dates <- lapply(summaries, `[[`, "dates")
  names(wealth) <- model_names
  names(dates) <- model_names

  all_dates <- as.Date(unique(unlist(lapply(dates, as.character))))
  all_dates <- sort(all_dates)

  plot(NA,
      xlim = range(all_dates), ylim = range(unlist(wealth)),
      xlab = "Date", ylab = "Portfolio wealth (start = 1)",
      main = "Volatility-timing strategy: cumulative wealth by model",
      xaxt = "n")
  axis.Date(1, at = pretty(all_dates, n = 8), format = "%Y-%m")

  is_nnsv <- model_names == "nonlinearSV"
  is_bh <- model_names == "BuyAndHold"
  for (i in seq_along(wealth)) {
    if (is_nnsv[i]) {
      lines(as.Date(dates[[i]]), wealth[[i]], col = "black", lwd = 2.5, lty = 1)
    } else if (is_bh[i]) {
      lines(as.Date(dates[[i]]), wealth[[i]], col = "black", lwd = 1.3, lty = 2)
    } else {
      lines(as.Date(dates[[i]]), wealth[[i]], col = "grey60", lwd = 1, lty = 1)
    }
  }
  legend("topleft",
        legend = c("nonlinear SV", "Buy-and-hold", "Other fitted models"),
        col = c("black", "black", "grey60"),
        lwd = c(2.5, 1.3, 1), lty = c(1, 2, 1),
        bty = "n", cex = 0.8)
}

# ---- 5. Rolling realized volatility vs. target ------------------------------
# 2026-08-27, added at the advisor's request: the figure meant to show
# whether each vol-targeting strategy actually delivers ~15% annualized vol
# (the SV models) or runs hotter than promised (the GARCH models). Grouped
# by model family so that pattern is visible at a glance rather than buried
# in seven similarly-styled lines: SV models in blue shades, GARCH models in
# red/orange shades, HAR-RV (neither) in grey, buy-and-hold's own realized
# vol as a thin dashed grey reference (it is not vol-targeted, so it is not
# expected to track the line -- included only to show how much a variance
# forecast is or is not damping exposure relative to doing nothing).
#
# window_weeks = 26 (roughly six months) trades off two things: shorter
# windows are noisier (a realized-vol estimate from a handful of weekly
# returns has a wide sampling distribution) and start later relative windows
# leave fewer rolling points inside the 153-week test period. 26 weeks
# leaves 128 rolling points and enough smoothing for the SV-vs-GARCH gap to
# read cleanly; adjust here if a different horizon is wanted.
rolling_annualized_vol <- function(portfolio_return, window_weeks = 26L) {
  n <- length(portfolio_return)
  out <- rep(NA_real_, n)
  if (n < window_weeks) return(out)
  for (i in window_weeks:n) {
    out[i] <- stats::sd(portfolio_return[(i - window_weeks + 1L):i]) * sqrt(52)
  }
  out
}

plot_rolling_volatility <- function(summaries, file,
                                    target_annual_vol = 0.15,
                                    window_weeks = 26L) {
  pdf(file, width = 9, height = 6)
  on.exit(dev.off())

  model_names <- vapply(summaries, function(x) x$row$Model, character(1))
  dates <- lapply(summaries, `[[`, "dates")
  roll <- lapply(summaries, function(x) rolling_annualized_vol(x$portfolio_return, window_weeks))
  names(roll) <- model_names
  names(dates) <- model_names

  sv_models <- c("linearSV", "regimeSwitchingSV", "nonlinearSV")
  garch_models <- c("sGARCH", "gjrGARCH", "eGARCH")
  sv_colors <- c(linearSV = "steelblue1", regimeSwitchingSV = "royalblue3",
                nonlinearSV = "navy")
  garch_colors <- c(sGARCH = "goldenrod1", gjrGARCH = "darkorange2",
                    eGARCH = "firebrick3")

  style <- function(m) {
    if (m %in% sv_models) list(col = sv_colors[[m]], lwd = 2, lty = 1)
    else if (m %in% garch_models) list(col = garch_colors[[m]], lwd = 2, lty = 1)
    else if (m == "harRV") list(col = "grey40", lwd = 2, lty = 1)
    else list(col = "grey70", lwd = 1, lty = 2)  # BuyAndHold
  }

  all_dates <- as.Date(unique(unlist(lapply(dates, as.character))))
  all_dates <- sort(all_dates)
  y_max <- max(target_annual_vol, unlist(roll), na.rm = TRUE) * 1.05

  plot(NA,
      xlim = range(all_dates), ylim = c(0, y_max),
      xlab = "Date", ylab = "Rolling annualized volatility",
      main = sprintf("%d-week rolling realized volatility vs. %.0f%% target",
                      window_weeks, 100 * target_annual_vol),
      xaxt = "n")
  axis.Date(1, at = pretty(all_dates, n = 8), format = "%Y-%m")
  abline(h = target_annual_vol, col = "black", lwd = 2, lty = 2)

  for (m in model_names) {
    s <- style(m)
    lines(as.Date(dates[[m]]), roll[[m]], col = s$col, lwd = s$lwd, lty = s$lty)
  }
  legend("topright",
        legend = c("15% target", sv_models, garch_models, "harRV", "BuyAndHold"),
        col = c("black", sv_colors[sv_models], garch_colors[garch_models], "grey40", "grey70"),
        lwd = c(2, rep(2, 6), 2, 1),
        lty = c(2, rep(1, 7), 2),
        bty = "n", cex = 0.7, ncol = 2)
}

# ---- 6. Pipeline entrypoint -------------------------------------------------

run_economic_results_pipeline <- function(
  results_file = "results/models/benchmark_results_all_available.rds",
  output_dir = "results/economic",
  target_annual_vol = 0.15,
  max_leverage = 2,
  risk_aversion = 3,
  bench = NULL
) {
  if (!file.exists(results_file)) {
    stop("Run run_models.R first — missing ", results_file, call. = FALSE)
  }
  benchmark_results <- readRDS(results_file)

  summaries <- lapply(names(benchmark_results), function(model_name) {
    model_economic_summary(
      model_name, benchmark_results[[model_name]],
      target_annual_vol, max_leverage, risk_aversion
    )
  })
  summaries <- Filter(Negate(is.null), summaries)
  if (!length(summaries)) {
    stop("No model in ", results_file, " has test-period forecasts.", call. = FALSE)
  }

  # Buy-and-hold uses whichever model's forecast_table is available first --
  # actual_return over the test split is identical across every model (same
  # underlying asset, same dates), so any of them gives the same row.
  bh <- buy_and_hold_summary(benchmark_results[[names(benchmark_results)[1]]],
                             target_annual_vol, risk_aversion, bench = bench)
  summaries <- c(summaries, list(bh))

  summary_table <- do.call(rbind, lapply(summaries, `[[`, "row"))
  summary_table <- summary_table[order(-summary_table$Sharpe), ]
  row.names(summary_table) <- NULL

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(summary_table, file.path(output_dir, "economic_summary.csv"), row.names = FALSE)
  saveRDS(list(summary = summary_table, models = summaries),
          file.path(output_dir, "economic_results.rds"), compress = TRUE)
  plot_wealth_paths(summaries, file.path(output_dir, "wealth_paths.pdf"))
  plot_rolling_volatility(
    summaries,  # all seven fitted models plus BuyAndHold
    file.path(output_dir, "rolling_volatility.pdf"),
    target_annual_vol = target_annual_vol
  )

  cat("\nEconomic significance (volatility-timing portfolio, test period)\n\n")
  print(summary_table, row.names = FALSE, digits = 3)
  cat("\nSaved to", output_dir, "\n")

  summary_table
}

if (isTRUE(getOption("benchmark.economic_autorun", TRUE))) {
  economic_results <- run_economic_results_pipeline()
}
