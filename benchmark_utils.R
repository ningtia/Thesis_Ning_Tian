default_benchmark_target <- function() {
  "y_IQQH_EUR"
}

default_benchmark_covariates <- function() {
  c(
    "GCPU_baseline",
    "l_t",
    "c_t",
    "itraxx_robustness",
    "Term_Spread",
    "Brent_return",
    "TTF_return",
    "log_VSTOXX",
    "COVID_dummy",
    "Energy_crisis_dummy"
  )
}

default_dummy_covariates <- function() {
  c("COVID_dummy", "Energy_crisis_dummy")
}

qlike <- function(actual_variance, forecast_variance) {
  eps <- .Machine$double.eps
  actual_variance <- pmax(as.numeric(actual_variance), eps)
  forecast_variance <- pmax(as.numeric(forecast_variance), eps)
  mean(actual_variance / forecast_variance - log(actual_variance / forecast_variance) - 1,
       na.rm = TRUE)
}

read_benchmark_raw <- function(train_file = "train_dataset.csv",
                               valid_file = "valid_dataset.csv",
                               test_file = "test_dataset.csv") {
  read_split <- function(path, split) {
    data <- read.csv(path, stringsAsFactors = FALSE)
    data$Date <- as.Date(data$Date)
    data$.split <- split
    data$.split_row <- seq_len(nrow(data))
    data
  }

  raw <- rbind(
    read_split(train_file, "train"),
    read_split(valid_file, "valid"),
    read_split(test_file, "test")
  )
}

standardize_covariates <- function(raw,
                                   covariates = default_benchmark_covariates(),
                                   dummy_covariates = default_dummy_covariates()) {
  scale_cols <- setdiff(covariates, dummy_covariates)
  scaling <- data.frame(
    variable = scale_cols,
    center = NA_real_,
    scale = NA_real_,
    stringsAsFactors = FALSE
  )

  train_rows <- raw$.split == "train"
  for (i in seq_along(scale_cols)) {
    col <- scale_cols[i]
    center <- mean(raw[train_rows, col], na.rm = TRUE)
    spread <- stats::sd(raw[train_rows, col], na.rm = TRUE)
    if (!is.finite(spread) || spread == 0) {
      spread <- 1
    }

    raw[[col]] <- (raw[[col]] - center) / spread
    scaling$center[i] <- center
    scaling$scale[i] <- spread
  }

  list(data = raw, scaling = scaling)
}

load_benchmark_data <- function(train_file = "train_dataset.csv",
                                valid_file = "valid_dataset.csv",
                                test_file = "test_dataset.csv",
                                target_col = default_benchmark_target(),
                                covariates = default_benchmark_covariates(),
                                dummy_covariates = default_dummy_covariates()) {
                                  
  raw <- read_benchmark_raw(train_file, valid_file, test_file)

  scaling <- NULL
  scaled <- standardize_covariates(raw, covariates, dummy_covariates)
  raw <- scaled$data
  scaling <- scaled$scaling
  x_cols <- covariates
  keep_cols <- c(target_col, x_cols)
  keep <- stats::complete.cases(raw[, keep_cols, drop = FALSE])
  data <- raw[keep, , drop = FALSE]

  X_all <- as.matrix(data[, x_cols, drop = FALSE])
  colnames(X_all) <- covariates
  y_all <- as.numeric(data[[target_col]])

  split_index <- split(seq_len(nrow(data)), data$.split)
  split_names <- c("train", "valid", "test")
  split_rows <- stats::setNames(
    lapply(split_names, function(split) {
      rows <- split_index[[split]]
    }),
    split_names
  )

  list(
    raw = raw,
    all = data,
    target_col = target_col,
    covariates = covariates,
    x_cols = x_cols,
    covariates_are_lagged = isTRUE(covariates_are_lagged),
    scaling = scaling,
    split_rows = split_rows,
    train = data[split_rows$train, , drop = FALSE],
    valid = data[split_rows$valid, , drop = FALSE],
    test = data[split_rows$test, , drop = FALSE],
    X_all = X_all,
    y_all = y_all,
    X_train = X_all[split_rows$train, , drop = FALSE],
    y_train = y_all[split_rows$train],
    X_valid = X_all[split_rows$valid, , drop = FALSE],
    y_valid = y_all[split_rows$valid],
    X_test = X_all[split_rows$test, , drop = FALSE],
    y_test = y_all[split_rows$test]
  )
}

actual_return_variance <- function(y) {
  as.numeric(y)^2
}

safe_forecast_variance <- function(x) {
  pmax(as.numeric(x), .Machine$double.eps)
}

make_oos_chunks <- function(bench,
                            refit_every = 13) {
  oos_indices <- c(bench$split_rows$valid, bench$split_rows$test)
  oos_indices <- sort(oos_indices)
  starts <- seq(1, length(oos_indices), by = refit_every)

  lapply(seq_along(starts), function(chunk_id) {
    start_pos <- starts[chunk_id]
    end_pos <- min(start_pos + refit_every - 1, length(oos_indices))
    forecast_indices <- oos_indices[start_pos:end_pos]
    train_end <- min(forecast_indices) - 1

    list(
      chunk_id = chunk_id,
      train_indices = seq_len(train_end),
      forecast_indices = forecast_indices,
      oos_positions = start_pos:end_pos,
      start_date = bench$all$Date[min(forecast_indices)],
      end_date = bench$all$Date[max(forecast_indices)]
    )
  })
}

split_oos_forecasts <- function(bench,
                                oos_indices,
                                forecast_variance) {
  forecast_variance <- safe_forecast_variance(forecast_variance)
  forecast_volatility <- sqrt(forecast_variance)
  split_label <- bench$all$.split[oos_indices]

  valid_mask <- split_label == "valid"
  test_mask <- split_label == "test"

  list(
    var_valid = forecast_variance[valid_mask],
    var_test = forecast_variance[test_mask],
    vol_valid = forecast_volatility[valid_mask],
    vol_test = forecast_volatility[test_mask],
    actual_valid = actual_return_variance(bench$all[[bench$target_col]][oos_indices[valid_mask]]),
    actual_test = actual_return_variance(bench$all[[bench$target_col]][oos_indices[test_mask]])
  )
}

stan_fit_diagnostics <- function(fit) {
  summary_fit <- summary(fit)$summary
  rhat <- summary_fit[, "Rhat"]
  bulk_ess <- summary_fit[, "n_eff"]

  sampler_params <- rstan::get_sampler_params(fit, inc_warmup = FALSE)
  divergence_sum <- sum(sapply(
    sampler_params,
    function(x) sum(x[, "divergent__"])
  ))

  list(
    max_rhat = max(rhat, na.rm = TRUE),
    min_bulk_ess = min(bulk_ess, na.rm = TRUE),
    rhat = rhat,
    bulk_ess = bulk_ess,
    divergence_sum = divergence_sum
  )
}
