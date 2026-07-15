source("benchmark_utils.R")

bench <- load_benchmark_data(
  standardize = TRUE,
  covariates_are_lagged = TRUE
)

find_realized_variance_column <- function(data, target_col) {
  target_suffix <- sub("^y_", "", target_col)
  candidates <- c(
    paste0("RV_", target_suffix),
    paste0("rv_", target_suffix),
    paste0("realized_variance_", target_suffix),
    "realized_variance",
    "RealizedVariance",
    "RV",
    "rv"
  )

  hit <- intersect(candidates, names(data))
  if (length(hit) == 0) {
    NULL
  } else {
    hit[1]
  }
}

lagged_mean <- function(x, window) {
  out <- rep(NA_real_, length(x))
  for (i in seq_along(x)) {
    if (i > window) {
      out[i] <- mean(x[(i - window):(i - 1)], na.rm = FALSE)
    }
  }
  out
}

rv_col <- find_realized_variance_column(bench$all, bench$target_col)
rv_source <- if (is.null(rv_col)) {
  "squared_return_proxy"
} else {
  rv_col
}

model_data <- bench$all

if (is.null(rv_col)) {
  message(
    "No intraday realized variance column found; ",
    "HAR-RV is estimated with squared returns as a proxy."
  )
  model_data$.rv <- actual_return_variance(model_data[[bench$target_col]])
} else {
  model_data$.rv <- as.numeric(model_data[[rv_col]])
}

# Only the endogenous RV history is lagged here. The external covariates
# remain the already-lagged x_{t-1} columns from the processed CSV files.
model_data$.rv <- safe_forecast_variance(model_data$.rv)
model_data$.log_rv <- log(model_data$.rv)
model_data$.log_rv_lag_1 <- log(safe_forecast_variance(lagged_mean(model_data$.rv, 1)))
model_data$.log_rv_lag_4 <- log(safe_forecast_variance(lagged_mean(model_data$.rv, 4)))
model_data$.log_rv_lag_13 <- log(safe_forecast_variance(lagged_mean(model_data$.rv, 13)))

har_cols <- c(".log_rv_lag_1", ".log_rv_lag_4", ".log_rv_lag_13")
required_cols <- c(".log_rv", har_cols, bench$covariates)
model_data <- model_data[stats::complete.cases(model_data[, required_cols, drop = FALSE]), ]

train_fit <- model_data[model_data$.split == "train", , drop = FALSE]
valid_fit <- model_data[model_data$.split == "valid", , drop = FALSE]
test_fit <- model_data[model_data$.split == "test", , drop = FALSE]

har_formula <- stats::as.formula(
  paste(".log_rv ~", paste(c(har_cols, bench$covariates), collapse = " + "))
)

fit_har_rv <- stats::lm(har_formula, data = train_fit)
sigma2_log <- stats::sigma(fit_har_rv)^2

predict_variance <- function(fit, newdata, sigma2_log) {
  pred_log <- stats::predict(fit, newdata = newdata)
  safe_forecast_variance(exp(pred_log + 0.5 * sigma2_log))
}

var_valid <- predict_variance(fit_har_rv, valid_fit, sigma2_log)
var_test <- predict_variance(fit_har_rv, test_fit, sigma2_log)

if (!exists("benchmark_results")) {
  benchmark_results <- list()
}

benchmark_results$HAR_RV <- list(
  fit = fit_har_rv,
  rv_source = rv_source,
  intraday_rv_available = !is.null(rv_col),
  forecasts = list(
    forecast_valid = var_valid,
    forecast_test = var_test,
    var_valid = var_valid,
    var_test = var_test,
    vol_valid = sqrt(var_valid),
    vol_test = sqrt(var_test),
    qlike_valid = qlike(actual_return_variance(valid_fit[[bench$target_col]]), var_valid),
    qlike_test = qlike(actual_return_variance(test_fit[[bench$target_col]]), var_test)
  ),
  data = list(
    target_col = bench$target_col,
    covariates = bench$covariates,
    covariates_are_lagged = bench$covariates_are_lagged,
    scaling = bench$scaling,
    har_windows = c(1, 4, 13)
  )
)
