#install.packages("rugarch")
library(tidyverse)
library(rugarch)
data <- read_csv("final_dataset.csv", col_types = cols(
                              Date = col_date(format = "%Y-%m-%d")))

model_data <- data[, c(
  "GCPU_baseline",
  "c_t",
  "itraxx_robustness",
  "Term_Spread",
  "Brent_return",
  "TTF_return",
  "log_VSTOXX",
  "COVID_dummy",
  "Energy_crisis_dummy",
  "y_IQQH_EUR",
  "y_ICLN",
  "y_TAN",
  "y_Global_Clean_Index",
  "y_stoxx"
)]
model_data <- na.omit(model_data)
X_main <- as.matrix(model_data[, c(
  "GCPU_baseline",
  "c_t",
  "itraxx_robustness",
  "Term_Spread",
  "Brent_return",
  "TTF_return",
  "log_VSTOXX",
  "COVID_dummy",
  "Energy_crisis_dummy"
)])
y_assets <- c("y_IQQH_EUR", "y_ICLN", "y_TAN", "y_Global_Clean_Index", "y_stoxx")


# 1.GARCH(1,1) with covariates
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1), external.regressors = X_main),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std"
)
for (current_asset in y_assets) {
  cat("\n=======================================================\n")
  cat("current asset:",current_asset)
  cat("=======================================================\n")

  y_data <- model_data[[current_asset]]
  fit_result <- ugarchfit(spec = garch_spec, data = y_data,solver = "gosolnp")
  print(fit_result@fit$matcoef)
}
