# =====================================================================
# 04_economic_figures.R
#
# The two economic figures for Chapter 8. Reads the series written by
# 03_conditional_economics.R; draws nothing that is not in those CSVs.
#
#   Figure 1  wealth paths -- proposed model in thick black, benchmarks in
#             thin grey, buy-and-hold dashed as the passive reference.
#   Figure 2  13-week rolling realised volatility of each strategy against
#             the 15% target the strategies were all asked to deliver.
# =====================================================================

if (dir.exists("Rlib")) .libPaths(c(normalizePath("Rlib"), .libPaths()))
ECON <- "results/economic"
dir.create(ECON, showWarnings = FALSE, recursive = TRUE)

TARGET <- 0.15
FOCUS  <- "nonlinearSV"
PASSIVE <- "buyAndHold"

wealth <- read.csv(file.path(ECON, "wealth_paths.csv"))
rollv  <- read.csv(file.path(ECON, "rolling_strategy_volatility.csv"))
wealth$date <- as.Date(wealth$date)
rollv$date  <- as.Date(rollv$date)

pretty_name <- c(sGARCH = "GARCH-X", gjrGARCH = "GJR-GARCH-X", eGARCH = "EGARCH-X",
                 harRV = "HAR-RV", linearSV = "Linear SV",
                 regimeSwitchingSV = "MS-SV", nonlinearSV = "NN-SV",
                 buyAndHold = "Buy and hold")

date_axis <- function(d) {
  ticks <- seq(as.Date(format(min(d), "%Y-01-01")), max(d), by = "6 months")
  axis(1, at = ticks, labels = format(ticks, "%Y-%m"), cex.axis = 0.8)
}

# ---------------------------------------------------------------------
# Figure 1: wealth paths
# ---------------------------------------------------------------------
models <- setdiff(names(wealth), c("date", FOCUS, PASSIVE))
pdf(file.path(ECON, "wealth_paths.pdf"), width = 8, height = 5)
par(mar = c(4.2, 4.2, 2.2, 1.0))
ylim <- range(unlist(wealth[-1]), na.rm = TRUE)
plot(wealth$date, wealth[[FOCUS]], type = "n", ylim = ylim,
     xaxt = "n", xlab = "", ylab = "Cumulative wealth (start = 1)",
     main = "Volatility-targeting strategies, test sample")
abline(h = 1, col = "grey80", lty = 3)
for (m in models) lines(wealth$date, wealth[[m]], col = "grey65", lwd = 1)
lines(wealth$date, wealth[[PASSIVE]], col = "grey25", lwd = 1.6, lty = 2)
lines(wealth$date, wealth[[FOCUS]], col = "black", lwd = 3)
date_axis(wealth$date)
legend("bottomleft", bty = "n", cex = 0.85,
       legend = c(pretty_name[[FOCUS]], pretty_name[[PASSIVE]],
                  "Five benchmark models"),
       col = c("black", "grey25", "grey65"),
       lwd = c(3, 1.6, 1), lty = c(1, 2, 1))
dev.off()

# ---------------------------------------------------------------------
# Figure 2: rolling realised volatility against the target
# ---------------------------------------------------------------------
sv  <- c("nonlinearSV", "regimeSwitchingSV", "linearSV")
gar <- c("sGARCH", "gjrGARCH", "eGARCH", "harRV")

pdf(file.path(ECON, "rolling_volatility.pdf"), width = 8, height = 5)
par(mar = c(4.2, 4.2, 2.2, 1.0))
keep <- !is.na(rollv[[FOCUS]])
ylim <- range(unlist(rollv[keep, setdiff(names(rollv), c("date", PASSIVE))]),
              TARGET, na.rm = TRUE)
plot(rollv$date[keep], rollv[[FOCUS]][keep], type = "n", ylim = ylim,
     xaxt = "n", xlab = "", ylab = "13-week realised volatility (annualised)",
     main = "Did the strategies deliver the volatility they promised?")
for (m in gar) lines(rollv$date[keep], rollv[[m]][keep], col = "grey70", lwd = 1)
for (m in setdiff(sv, FOCUS)) lines(rollv$date[keep], rollv[[m]][keep],
                                    col = "grey40", lwd = 1.3)
lines(rollv$date[keep], rollv[[FOCUS]][keep], col = "black", lwd = 3)
abline(h = TARGET, col = "firebrick", lwd = 2, lty = 2)
date_axis(rollv$date[keep])
legend("topright", bty = "n", cex = 0.85,
       legend = c(pretty_name[[FOCUS]], "Other SV models",
                  "GARCH-type and HAR", "15% target"),
       col = c("black", "grey40", "grey70", "firebrick"),
       lwd = c(3, 1.3, 1, 2), lty = c(1, 1, 1, 2))
dev.off()

cat("wrote", file.path(ECON, "wealth_paths.pdf"), "and",
    file.path(ECON, "rolling_volatility.pdf"), "\n")

# quick numeric readout that the figures are meant to make visible
above <- sapply(setdiff(names(rollv), "date"),
                function(m) mean(rollv[[m]] > TARGET, na.rm = TRUE))
cat("\nshare of weeks with 13-week realised vol above the 15% target:\n")
print(round(sort(above), 3))
