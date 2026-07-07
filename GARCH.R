#install.packages("rugarch")
library(tidyverse)
library(rugarch)
train <- read_csv("train_dataset.csv", col_types = cols(
  Date = col_date(format = "%Y-%m-%d")))
valid <- read_csv("valid_dataset.csv", col_types = cols(
  Date = col_date(format = "%Y-%m-%d")))
test <- read_csv("test_dataset.csv", col_types = cols(
  Date = col_date(format = "%Y-%m-%d")))

n_train <- nrow(train)
n_valid <- nrow(valid)
n_test <- nrow(test)

cols <- c("GCPU_baseline",
          "c_t",
          "itraxx_robustness",
          "Term_Spread",
          "Brent_return",
          "TTF_return",
          "log_VSTOXX")
train[cols] <- scale(train[cols])
valid[cols] <- scale(valid[cols])
test[cols] <- scale(test[cols])

total_data <- rbind(train, valid, test)
model_data <- total_data[, c(
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
y_data = model_data[["y_IQQH_EUR"]]

# 1.GARCH(1,1) with covariates
spec1 <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1), external.regressors = X_main),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "std"
)
# train the model with out-of-sample data for validation and testing
fit1 <- ugarchfit(spec = spec1, data = y_data, out.sample = c(n_valid + n_test))
print(fit1@fit$matcoef)
persistence(fit1)
# valid and test the model
fc_sgarch <- ugarchforecast(fit1, n.ahead = 1, n.roll = (n_valid + n_test))
all_pred_sigma1 <- as.numeric(sigma(fc_sgarch))
valid_pred_sigma1 <- (all_pred_sigma1[1:n_valid])^2
test_pred_sigma1 <- (all_pred_sigma1[(n_valid + 1):(n_valid + n_test)])^2

# 2. GJR-GARCH(1,1) with covariates
spec2 <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1), external.regressors = X_main),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "std"
)
# train the model with out-of-sample data for validation and testing
fit2 <- ugarchfit(spec = spec2, data = y_data, out.sample = c(n_valid + n_test))
print(fit2@fit$matcoef)
persistence(fit2)
# valid and test the model
fc_gjr <- ugarchforecast(fit2, n.ahead = 1, n.roll = (n_valid + n_test))
all_pred_sigma2 <- as.numeric(sigma(fc_gjr))
valid_pred_sigma2 <- (all_pred_sigma2[1:n_valid])^2
test_pred_sigma2 <- (all_pred_sigma2[(n_valid + 1):(n_valid + n_test)])^2

# 3. EGARCH(1,1) with covariates
spec3 <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1), external.regressors = X_main),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "std"
)
fit3 <- ugarchfit(spec = spec3, data = y_data, solver = "gosolnp", out.sample = c(n_valid + n_test))
print(fit3@fit$matcoef)
persistence(fit3)
# valid and test the model
fc_egarch <- ugarchforecast(fit3, n.ahead = 1, n.roll = (n_valid + n_test))
all_pred_sigma3 <- as.numeric(sigma(fc_egarch))
valid_pred_sigma3 <- (all_pred_sigma3[1:n_valid])^2
test_pred_sigma3 <- (all_pred_sigma3[(n_valid + 1):(n_valid + n_test)])^2

