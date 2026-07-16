# install.packages("rugarch")
library(rugarch)
library(xts)

source("benchmark_utils.R")

bench <- load_benchmark_data()

n_train <- nrow(bench$train)
n_valid <- nrow(bench$valid)
n_test <- nrow(bench$test)
n_out <- n_valid + n_test

X_main <- xts(
  bench$X_all,
  order.by = bench$all$Date
)

y_data <- xts(
  bench$y_all,
  order.by = bench$all$Date
)

fit_garch_benchmark <- function(model_name, solver = "hybrid") {
  spec <- ugarchspec(
    variance.model = list(
      model = model_name,
      garchOrder = c(1, 1),
      external.regressors = as.matrix(X_main)
    ),
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
    distribution.model = "std"
  )

  fit <- ugarchfit(
    spec = spec,
    data = y_data,
    solver = solver,
    out.sample = n_out
  )

  fc <- ugarchforecast(
    fit,
    n.ahead = 1,
    n.roll = n_out - 1
  )

  all_pred_var <- safe_forecast_variance(as.numeric(sigma(fc))^2)
  valid_pred_var <- all_pred_var[seq_len(n_valid)]
  test_pred_var <- all_pred_var[n_valid + seq_len(n_test)]

  list(
    model_name = model_name,
    fit = fit,
    conditional_sigma = sigma(fit),
    coef = fit@fit$matcoef,
    converged = fit@fit$convergence,
    persistence = persistence(fit),
    forecasts = list(
      forecast_valid = valid_pred_var,
      forecast_test = test_pred_var,
      var_valid = valid_pred_var,
      var_test = test_pred_var,
      vol_valid = sqrt(valid_pred_var),
      vol_test = sqrt(test_pred_var),
      qlike_valid = qlike(actual_return_variance(bench$y_valid), valid_pred_var),
      qlike_test = qlike(actual_return_variance(bench$y_test), test_pred_var)
    ),
    data = list(
      target_col = bench$target_col,
      covariates = bench$covariates,
      covariates_are_lagged = bench$covariates_are_lagged,
      scaling = bench$scaling
    )
  )
}

if (!exists("benchmark_results")) {
  benchmark_results <- list()
}

benchmark_results$sGARCH <- fit_garch_benchmark("sGARCH")
benchmark_results$gjrGARCH <- fit_garch_benchmark("gjrGARCH")
benchmark_results$eGARCH <- fit_garch_benchmark("eGARCH",solver = "gosolnp")
