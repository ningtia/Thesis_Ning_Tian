# ==============================================================================
# NN-SV 敏感性分析:tau_w_scale(K、s 固定)
#
# 老师的原话:"the nonlinearity is now shrunk twice: s is fixed at 0.5 (it was
# a parameter before), AND the tau_w prior is now very tight (0.2). Together,
# these can make the network too weak, even if there is a real nonlinear
# effect in the data. Please run a sensitivity check: tau_w prior scale in
# {0.2, 0.5, 1.0} and s in {0.25, 0.5, 1.0}."
#
# 2026-08-23 改成 s 固定、只测 tau_w_scale:recovery_replications.R 那边单独
# 测过"s: 0.5 -> 1.0,tau_w_scale 不变",结果 divergence 明显变多(main/zero/
# strong 的 usable_rate 全面下滑)——因为 g = s * tanh(...) 里 tanh 已经饱和
# 在 [-1,1],s 只是放大这个饱和值的倍数,s 越大、tau_w_scale 不跟着收紧的话,
# 网络能注入 h_t 的冲击就越大,geometry 越难采。所以 s 和 tau_w_scale 不能
# 独立调,必须联合看——这里固定 s = SENS_S_FIXED(见下),只扫 tau_w_scale,
# 就是为了单独看"tau_w_scale 多紧,才能在 s=1.0 时把 divergence 压回去"。
#
# 2026-08-24 K 也不测了,固定在生产架构 default_nn_candidates() 用的
# K = 8L(见 nonlinear_sv.R)。老师原话里从没要求扫 K,那是这份脚本自己加
# 的额外维度;既然 s 已经固定、只关心 tau_w_scale 这一个旋钮,K 继续当网格
# 维度只是多花 3 倍算力,不会多回答任何问题——网格缩到 scenario x
# tau_w_scale,6 格。
#
# 老师也说了 simulation 在 strong case 下没问题就够了,不需要一定用真实数据
# 磨合,所以整个检查跑在已知真值的模拟数据上,直接复用
# results/simulation/ 已经生成好的数据集,不重新模拟、不碰真实 benchmark 数据:
#   - main:   nnsv_simulation_main.csv + nnsv_main_true_parameters.csv
#             (simulation_study_nnsv.R 的设计 (a),真值已存成 csv)
#   - strong: nnsv_simulation_strong.csv(设计 (c))——没有单独的
#             true_parameters 文件,真值按该脚本注释里写的方式现算:
#             mu = 1.80 + mean(g_true[2:T])/(1-0.95),g_sd = sd(g_true[2:T])。
#
# 每个 (scenario, tau_w_scale) 格子只拟合一次,不做多次复现:这是筛选
# 网格,只需要看出"收缩不收缩/发不发散"的方向和大小,不需要给全部 6 格都算
# 出可发表的 bias/coverage。选出赢家之后,要拿正式数字写进论文,再用
# recovery_replications.R 的复现引擎单独对那一格多次重跑。
# ==============================================================================
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
options(benchmark.nn_autorun = FALSE)
source("nonlinear_sv.R")
rstan_options(auto_write = TRUE)


# ------------------------------ 运行设置 ---------------------------------------

SENS_SIM_DIR    <- "results/simulation"
SENS_OUTPUT_DIR <- "results/nnsv_sensitivity"

SENS_SCENARIOS    <- c("main", "strong")
SENS_K_VALUES     <- 4L    # 固定值,不再是网格维度;见上方 2026-08-24 说明(= default_nn_candidates() 的生产 K)
SENS_TAU_W_SCALES <- c(0.2, 0.5, 1.0)
SENS_S_FIXED      <- 1.0   # 固定值,不再是网格维度;见上方 2026-08-23 说明


SENS_CHAINS        <- 4L
SENS_WARMUP        <- 2000L
SENS_SAMPLING      <- 4000L
SENS_ADAPT_DELTA   <- 0.99
SENS_MAX_TREEDEPTH <- 12L
SENS_FIT_SEED      <- 20260820L

sens_targets <- c("mu", "phi", "sigma_eta", "nu", "g_sd", "h_bar")


# ---------------------------------------------------------------------------
# 读一个场景的模拟数据 + 真值。main 有现成的 true_parameters csv;strong 没
# 有,就按 simulation_study_nnsv.R 生成 main_true_parameters 用的同一套公式
# 现算(该脚本注释里也是这么说的)。
# ---------------------------------------------------------------------------
load_sim_scenario <- function(label, sim_dir = SENS_SIM_DIR) {
  sim_data <- utils::read.csv(file.path(sim_dir, sprintf("nnsv_simulation_%s.csv", label)))
  T_sim <- nrow(sim_data)
  g <- sim_data$g_true[2:T_sim]

  true_params_path <- file.path(sim_dir, sprintf("nnsv_%s_true_parameters.csv", label))
  if (file.exists(true_params_path)) {
    tp <- utils::read.csv(true_params_path)
    get_true <- function(nm) tp$true_value[tp$parameter == nm]
    true_vec <- c(mu = get_true("mu"), phi = get_true("phi"),
                 sigma_eta = get_true("sigma_eta"), nu = get_true("nu"),
                 g_sd = get_true("g_sd"))
  } else {
    true_vec <- c(
      mu        = 1.80 + mean(g) / (1 - 0.95),
      phi       = 0.95,
      sigma_eta = 0.25,
      nu        = 10,
      g_sd      = sd(g)
    )
  }
  true_vec["h_bar"] <- mean(sim_data$h_true)

  list(sim_data = sim_data, T_sim = T_sim, true_vec = true_vec)
}

# ---------------------------------------------------------------------------
# 一个格子:一次拟合,记录标量恢复(inside_95/z/bias_frac ——单次拟合没有
# 跨复现的 coverage 率)+ g 形状恢复 + 完整拟合诊断。warm-start
# (nn_sv_linear_start/nn_sv_init)和诊断套件(stan_fit_diagnostics)都来自顶部
# source("nonlinear_sv.R")(它又 source 了 benchmark_utils.R)。
# ---------------------------------------------------------------------------
run_sens_cell <- function(compiled_model, label, scenario, K, tau_w_scale, seed) {
  sim_data <- scenario$sim_data
  T_sim    <- scenario$T_sim
  true_vec <- scenario$true_vec

  stan_data <- list(
    T = T_sim, D = 1, K = K,
    y = sim_data$y,
    X = matrix(sim_data$x[1:(T_sim - 1)], ncol = 1),
    use_student_t = 1,
    x_forecast = array(sim_data$x[T_sim], dim = 1),
    s_fixed = SENS_S_FIXED, tau_w_scale = tau_w_scale,
    mu_scale = 5, phi_a = 20, phi_b = 1.5,
    sigma_eta_scale = 1, b1_scale = 0.5, nu_rate = 0.1,
    prior_only = 0L, stationary_init = 0L, h1_scale = 2.0
  )

  linear_start <- nn_sv_linear_start(stan_data$y, stan_data$X)
  init_fn <- function() nn_sv_init(stan_data, previous_post = NULL, linear_start = linear_start)

  fit <- rstan::sampling(
    compiled_model, data = stan_data, chains = SENS_CHAINS, cores = SENS_CHAINS,
    iter = SENS_WARMUP + SENS_SAMPLING, warmup = SENS_WARMUP, seed = seed,
    refresh = 0, init = init_fn,
    control = list(adapt_delta = SENS_ADAPT_DELTA, max_treedepth = SENS_MAX_TREEDEPTH)
  )

  diagnostics <- stan_fit_diagnostics(fit, pars = c(sens_targets, "tau_w"))

  dr <- rstan::extract(fit, pars = sens_targets)
  recovery <- do.call(rbind, lapply(sens_targets, function(p) {
    v   <- as.numeric(dr[[p]])
    med <- median(v); s <- sd(v)
    lo  <- quantile(v, 0.025, names = FALSE); hi <- quantile(v, 0.975, names = FALSE)
    tv  <- unname(true_vec[p])
    data.frame(
      scenario = label, K = K, tau_w_scale = tau_w_scale,
      parameter = p, true_value = tv,
      posterior_median = med, posterior_sd = s, lower_95 = lo, upper_95 = hi,
      inside_95 = tv >= lo & tv <= hi,
      bias      = med - tv,
      bias_frac = if (tv != 0) (med - tv) / tv else NA_real_,
      z         = (med - tv) / s
    )
  }))

  true_g_c <- sim_data$g_true[2:T_sim] - mean(sim_data$g_true[2:T_sim])
  g_draws  <- rstan::extract(fit, pars = "g")$g
  g_median <- apply(g_draws, 2, median)
  shape <- data.frame(
    scenario = label, K = K, tau_w_scale = tau_w_scale,
    correlation = if (sd(true_g_c) == 0) NA_real_ else cor(g_median, true_g_c),
    rmse        = sqrt(mean((g_median - true_g_c)^2))
  )

  attempt <- data.frame(
    scenario = label, K = K, tau_w_scale = tau_w_scale,
    divergences  = diagnostics$divergence_sum,
    max_rhat     = diagnostics$max_rhat,
    min_ess_bulk = diagnostics$min_bulk_ess,
    min_ess_tail = diagnostics$min_tail_ess,
    min_ebfmi    = diagnostics$min_ebfmi,
    passes_gate  = isTRUE(diagnostics$all_checks_passed)
  )

  message(sprintf("[%s K=%d tw=%.2f s=%.2f] divergences=%d, passes_gate=%s",
                  label, K, tau_w_scale, SENS_S_FIXED, diagnostics$divergence_sum,
                  isTRUE(diagnostics$all_checks_passed)))
  rm(fit); invisible(gc())

  list(recovery = recovery, shape = shape, attempt = attempt)
}

# ---------------------------------------------------------------------------
# 完整网格:scenarios x tau_w_scale(K 固定在 SENS_K_VALUES,s 固定在
# SENS_S_FIXED),每格一次拟合。写出三张明细表和一张"收缩读数"汇总表。
# 不自动选赢家——见文件末尾的
# 读表说明。
# ---------------------------------------------------------------------------
run_sens_grid <- function(compiled_model,
                          scenarios = SENS_SCENARIOS,
                          K_values = SENS_K_VALUES,
                          tau_w_scales = SENS_TAU_W_SCALES,
                          output_dir = SENS_OUTPUT_DIR) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  loaded <- stats::setNames(lapply(scenarios, load_sim_scenario), scenarios)

  plan <- expand.grid(
    scenario = scenarios, K = K_values, tau_w_scale = tau_w_scales,
    stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE
  )

  cells <- lapply(seq_len(nrow(plan)), function(i) {
    run_sens_cell(
      compiled_model = compiled_model,
      label = plan$scenario[i], scenario = loaded[[plan$scenario[i]]],
      K = plan$K[i], tau_w_scale = plan$tau_w_scale[i],
      seed = SENS_FIT_SEED + i - 1L
    )
  })

  recovery_tbl <- do.call(rbind, lapply(cells, `[[`, "recovery"))
  shape_tbl    <- do.call(rbind, lapply(cells, `[[`, "shape"))
  attempt_tbl  <- do.call(rbind, lapply(cells, `[[`, "attempt"))

  write.csv(recovery_tbl, file.path(output_dir, "ks_sensitivity_recovery.csv"), row.names = FALSE)
  write.csv(shape_tbl,    file.path(output_dir, "ks_sensitivity_shape.csv"), row.names = FALSE)
  write.csv(attempt_tbl,  file.path(output_dir, "ks_sensitivity_attempts.csv"), row.names = FALSE)

  # -----------------------------------------------------------------------
  # 收缩读数:只挑 g_sd 这一行(老师关切的对象),把 bias_frac、收敛 gate、
  # g 形状相关性并成一张表,一眼能看全。
  # -----------------------------------------------------------------------
  g_sd_rows <- recovery_tbl[recovery_tbl$parameter == "g_sd",
                            c("scenario", "K", "tau_w_scale",
                             "true_value", "posterior_median", "bias", "bias_frac", "inside_95")]
  shrinkage_tbl <- merge(g_sd_rows, attempt_tbl, by = c("scenario", "K", "tau_w_scale"))
  shrinkage_tbl <- merge(shrinkage_tbl,
                         shape_tbl[, c("scenario", "K", "tau_w_scale", "correlation", "rmse")],
                         by = c("scenario", "K", "tau_w_scale"))
  shrinkage_tbl <- shrinkage_tbl[order(shrinkage_tbl$scenario, shrinkage_tbl$K,
                                       shrinkage_tbl$tau_w_scale), ]
  write.csv(shrinkage_tbl, file.path(output_dir, "ks_shrinkage_readout.csv"), row.names = FALSE)
  print(shrinkage_tbl)

  list(recovery = recovery_tbl, shape = shape_tbl, attempts = attempt_tbl,
       shrinkage_readout = shrinkage_tbl)
}

# 读 ks_shrinkage_readout.csv 的方法:s 和 K 现在都固定了,只在每个
# scenario 内看 bias_frac 是不是随 tau_w_scale 收紧到 0.2 而系统性
# 变得更负——如果 main 和 strong 都有这个趋势,是"缩太狠"的证据。但这次更
# 关键的是 passes_gate:s 固定在 1.0 之后,tau_w_scale 越松,divergence 可能
# 越多(见文件头 2026-08-23 的说明)——所以先看哪些 tau_w_scale 能让
# passes_gate 稳定为 TRUE,再在这些格子里挑 |bias_frac| 小、两个场景下
# correlation 都接近 1 的。passes_gate 不过的格子,不管 bias 数字好不好看
# 都不能信——那是没收敛的后验。
#
# 这是单次拟合的筛选网格,不是可发表的复现研究:
#   1. 选出赢家后,要拿正式的 bias/RMSE/coverage 数字,用
#      recovery_replications.R 的复现引擎对那一格单独多次重跑。
#   2. 这里没有测 zero 场景 —— 放松 tau_w_scale 修"缩太狠"的同时,可能把
#      之前验过的假阳性控制又打开了,赢家选出来后要单独在 zero 场景
#      (nnsv_simulation_null.csv)上验一次,不测这一步,修复就没有被验证过。


# ==============================================================================
# 主程序
# ==============================================================================

if (isTRUE(getOption("benchmark.nn_sensitivity_autorun", TRUE))) {
  config <- nn_sv_default_config()
  config$chains <- SENS_CHAINS
  configure_nn_sv_parallel(config$chains)
  # config$stan_file 默认就是生产用的 nonlinear_sv.stan(已确认和
  # results/simulation/nonlinear_sv_v3.stan 架构一致)。
  compiled_model <- rstan::stan_model(file = config$stan_file)

  nnsv_sensitivity_result <- run_sens_grid(compiled_model = compiled_model)
}
