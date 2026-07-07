library(rstan)

X <- as.matrix(
  train[, c(
    "GCPU_baseline",
    "c_t",
    "Brent_return",
    "TTF_return",
    "log_VSTOXX",
    "COVID_dummy",
    "Energy_crisis_dummy"
  )]
)

Y <- as.numeric(train$y_IQQH_EUR)

stan_train <- list(
T=nrow(X),
K=ncol(X),
y=Y,
X=X
)

###########################
# Model 4 : Linear Stochastic Volatility (Stan)
###########################

fit_linear <- stan(
file="linear_sv.stan",
data=stan_train,
chains=4,
iter=4000,
warmup=2000,
seed=666
)
post_linear <- rstan::extract(fit_linear)







benchmark_results$linearSV <- list(
  fit = fit_linear,
  posterior = post_linear,
  forecast_valid = pred_valid,
  forecast_test = pred_test,
  var_valid = var_valid,
  var_test = var_test,
  qlike_valid = qlike_valid,
  qlike_test = qlike_test
)
