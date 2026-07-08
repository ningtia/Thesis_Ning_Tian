library(rstan)

X_vars <- c(
  "GCPU_baseline",
  "c_t",
  "Brent_return",
  "TTF_return",
  "log_VSTOXX",
  "COVID_dummy",
  "Energy_crisis_dummy"
)
X <- as.matrix(
  train[, X_vars]
)

Y <- as.numeric(train$y_IQQH_EUR)

qlike <- function(actual,pred){
  mean(actual/pred-log(actual/pred)-1)
}

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

############################################################
## Validation + Test
############################################################
oos_data <- rbind(valid, test)
N_out <- nrow(oos_data)
X_out <- as.matrix(oos_data[, X_vars])
############################################################
## Prediction matrix
############################################################
var_draws <- matrix(NA, n_draws, N_out)
############################################################
## Initial latent state
############################################################
h_prev <- post$h[, nrow(train)]
############################################################
## Recursive forecast
############################################################
for(t in 1:N_out){
  xb <- as.vector(post$beta %*% X_out[t,])
  h_now <- post$mu + post$phi*(h_prev-post$mu) + xb
  var_draws[,t] <- exp(h_now + 0.5 * post$sigma_eta^2)
  h_prev <- h_now
}
############################################################
## Posterior mean
############################################################
forecast_variance <- colMeans(var_draws)
forecast_volatility <- sqrt(forecast_variance)
############################################################
## Split back
############################################################
n_valid <- nrow(valid)
var_valid <-
  forecast_variance[1:n_valid]
var_test <-
  forecast_variance[(n_valid+1):N_out]
vol_valid <-
  forecast_volatility[1:n_valid]
vol_test <-
  forecast_volatility[(n_valid+1):N_out]


qlike_valid <- qlike(actual_valid,var_valid)
qlike_test <- qlike(actual_test,var_test)

# diagnostic metrics
## r-hat Values close to 1 indicate convergence.
summary_fit <- summary(fit_linear)$summary
rhat <- summary_fit[, "Rhat"]
max(rhat)

## ess value over 400 indicates good mixing of the chains.
bulk_ess <- summary_fit[, "n_eff"]
min(bulk_ess)

## Divergence values close to 0 indicate good mixing of the chains.
sampler_params <- get_sampler_params(fit_linear, inc_warmup = FALSE)
divergence_sum <- sum(sapply(
  sampler_params,
  function(x) sum(x[, "divergent__"])
))


benchmark_results$linearSV <- list(
  fit = fit_linear,
  posterior = post_linear,
  forecasts = list(
    forecast_valid = pred_valid,
    forecast_test = pred_test,
    var_valid = var_valid,
    var_test = var_test,
    vol_valid = vol_valid,
    vol_test = vol_test,
    qlike_valid = qlike_valid,
    qlike_test = qlike_test),
  diagnostics = list(
    rhat = rhat,
    bulk_ess = bulk_ess,
    divergence_sum = divergence_sum
  )
)
