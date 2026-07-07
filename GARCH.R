#install.packages("rugarch")
library(tidyverse)
library(rugarch)
library(xts)

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
  "y_IQQH_EUR",
  "Date"
)]
model_data <- na.omit(model_data)
X_main <- xts(
  as.matrix(
  model_data[, c(
    "GCPU_baseline",
    "c_t",
    "Brent_return",
    "TTF_return",
    "log_VSTOXX",
    "COVID_dummy",
    "Energy_crisis_dummy"
  )]
),
  order.by = model_data$Date)
  
y_data <- xts(
  model_data$y_IQQH_EUR,
  order.by = model_data$Date
)

# 1.GARCH(1,1) with covariates
spec1 <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1), external.regressors = X_main),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "std"
)
# train the model with out-of-sample data for validation and testing
fit1 <- ugarchfit(spec = spec1, data = y_data, out.sample = c(n_valid + n_test))
sGARCH <- list()
sGARCH$fit <- fit1
sGARCH$conditional_sigma = sigma(fit1)
sGARCH$coef <- fit1@fit$matcoef
sGARCH$converged <- fit1@fit$convergence
sGARCH$persistence <- persistence(fit1)

# valid and test the model
fc_sgarch <- ugarchforecast(fit1, n.ahead = 1, n.roll = (n_valid + n_test)-1)
sGARCH$all_pred_sigma <- sigma(fc_sgarch)
sGARCH$valid_pred_sigma <- (sGARCH$all_pred_sigma[1:n_valid])^2
sGARCH$test_pred_sigma <- (sGARCH$all_pred_sigma[(n_valid + 1):(n_valid + n_test)])^2

# 2. GJR-GARCH(1,1) with covariates
spec2 <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1), external.regressors = X_main),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "std"
)
# train the model with out-of-sample data for validation and testing
fit2 <- ugarchfit(spec = spec2, data = y_data, out.sample = c(n_valid + n_test))
gjrGARCH <- list()
gjrGARCH$fit <- fit2
gjrGARCH$conditional_sigma = sigma(fit2)
gjrGARCH$coef <- fit2@fit$matcoef
gjrGARCH$converged <- fit2@fit$convergence
gjrGARCH$persistence <- persistence(fit2)
# valid and test the model
fc_gjr <- ugarchforecast(fit2, n.ahead = 1, n.roll = (n_valid + n_test)-1)
gjrGARCH$all_pred_sigma <- sigma(fc_gjr)
gjrGARCH$valid_pred_sigma <- (gjrGARCH$all_pred_sigma[1:n_valid])^2
gjrGARCH$test_pred_sigma <- (gjrGARCH$all_pred_sigma[(n_valid + 1):(n_valid + n_test)])^2

# 3. EGARCH(1,1) with covariates
spec3 <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1), external.regressors = X_main),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "std"
)
fit3 <- ugarchfit(spec = spec3, data = y_data, solver = "gosolnp", out.sample = c(n_valid + n_test))
eGARCH <- list()
eGARCH$fit <- fit3
eGARCH$conditional_sigma = sigma(fit3)
eGARCH$coef <- fit3@fit$matcoef
eGARCH$converged <- fit3@fit$convergence
eGARCH$persistence <- persistence(fit3)
# valid and test the model
fc_egarch <- ugarchforecast(fit3, n.ahead = 1, n.roll = (n_valid + n_test)-1)
eGARCH$all_pred_sigma <- sigma(fc_egarch)
eGARCH$valid_pred_sigma <- (eGARCH$all_pred_sigma[1:n_valid])^2
eGARCH$test_pred_sigma <- (eGARCH$all_pred_sigma[(n_valid + 1):(n_valid + n_test)])^2

benchmark_results <- list(
  sGARCH = sGARCH,
  gjrGARCH = gjrGARCH,
  eGARCH = eGARCH
)



