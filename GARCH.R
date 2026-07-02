#install.packages("rugarch")
library(tidyverse)
library(rugarch)
train <- read_csv("train_dataset.csv", col_types = cols(
  Date = col_date(format = "%Y-%m-%d")))
valid <- read_csv("valid_dataset.csv", col_types = cols(
  Date = col_date(format = "%Y-%m-%d")))
test <- read_csv("test_dataset.csv", col_types = cols(
  Date = col_date(format = "%Y-%m-%d")))

model_data <- train[, c(
  "GCPU_baseline",
  "c_t",
  "itraxx_robustness",
  "Term_Spread",
  "Brent_return",
  "TTF_return",
  "log_VSTOXX",
  "COVID_dummy",
  "Energy_crisis_dummy",
  "y_IQQH_EUR"
)]
model_data <- na.omit(model_data)
cols <- c("GCPU_baseline",
         "c_t",
         "itraxx_robustness",
         "Term_Spread",
         "Brent_return",
         "TTF_return",
         "log_VSTOXX")
model_data[cols] <- scale(model_data[cols])
X_main <- as.matrix(
  model_data[, c(
    "GCPU_baseline",
    "c_t",
    "Brent_return",
    "TTF_return",
    "log_VSTOXX",
    "COVID_dummy",
    "Energy_crisis_dummy"
  )]
)

# 1.GARCH(1,1) with covariates
spec1 <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1), external.regressors = X_main),
  mean.model = list(armaOrder = c(0, 0), external.regressors = X_main),
  distribution.model = "std"
)
y_data = model_data[["y_IQQH_EUR"]]
fit1 <- ugarchfit(spec = spec1, data = y_data, solver = "gosolnp")
print(fit1@fit$matcoef)

