################################################################################
## 文件名: evaluate_results.R
## 功能: 补充实现论文第 4, 5, 10 节缺失的全部诊断、回测、以及经济显著性评估
################################################################################

library(tidyverse)
library(rstan)
library(forecast)    # 用于 DM 检验
library(zoo)         # 用于最大回撤计算

# 确保加载了基础工具包
source("benchmark_utils.R")

# 假设主流程运行完毕后，你的内存中存在 bench 对象和 benchmark_results 列表
# 如果是读取保存的文件，可以取消下行注释：
# load("all_benchmark_results.RData")

# 获取测试集的真实收益率和真实方差代理
y_test <- bench$y_test
actual_var_test <- actual_return_variance(y_test)
N_test <- length(y_test)

cat("========================================================================\n")
cat(" 步骤 4 补充: 贝叶斯收敛诊断 (针对 Stan 模型)\n")
cat("========================================================================\n")

check_stan_diagnostics <- function(fit_obj, model_name) {
  if (is.null(fit_obj)) return(NULL)

  cat("\n[诊断报告] 模型:", model_name, "\n")

  # 1. Rhat 与 Bulk-ESS
  summary_fit <- summary(fit_obj)$summary
  max_rhat <- max(summary_fit[, "Rhat"], na.rm = TRUE)
  min_ess <- min(summary_fit[, "n_eff"], na.rm = TRUE)

  cat("  - 最大 Rhat (要求 < 1.01):", round(max_rhat, 4),
      if(max_rhat < 1.01) "【合格】" else "【不合格，需增加iter】", "\n")
  cat("  - 最小 Bulk-ESS (要求 > 400):", round(min_ess, 1),
      if(min_ess > 400) "【合格】" else "【不合格】", "\n")

  # 2. 发散跳跃 (Divergent Transitions)
  sampler_params <- get_sampler_params(fit_obj, inc_warmup = FALSE)
  div_sum <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
  cat("  - 发散跳跃总数 (要求接近 0):", div_sum,
      if(div_sum == 0) "【完美】" else "【警告：需提高 adapt_delta】", "\n")

  # 3. 补充论文缺失的 E-BFMI 能量诊断
  bfmi <- rstan::get_bfmi(fit_obj)
  min_bfmi <- min(bfmi)
  cat("  - 最小 E-BFMI (要求 > 0.3):", round(min_bfmi, 4),
      if(min_bfmi > 0.3) "【合格】" else "【警告：能量通道不佳】", "\n")

  # 4. 自动保存可视化 Check 图形 (迹线图与关键参数成对图)
  # 自动识别模型里包含哪些参数
  available_pars <- fit_obj@model_pars
  key_pars <- intersect(c("phi", "sigma_eta", "nu", "p11", "p22"), available_pars)

  if (length(key_pars) > 0) {
    pdf(paste0("diagnostics_", model_name, ".pdf"), width = 10, height = 7)
    print(stan_trace(fit_obj, pars = key_pars))
    if (length(key_pars) >= 2) {
      print(pairs(fit_obj, pars = key_pars[1:2]))
    }
    dev.off()
    cat("  - 可视化检查图形已保存至:", paste0("diagnostics_", model_name, ".pdf"), "\n")
  }
}

# 运行 Stan 诊断
if (!is.null(benchmark_results$linearSV)) {
  check_stan_diagnostics(benchmark_results$linearSV$fit, "Linear_SV")
}
if (!is.null(benchmark_results$regimeSwitchingSV)) {
  check_stan_diagnostics(benchmark_results$regimeSwitchingSV$fit, "Regime_Switching_SV")
}


cat("\n========================================================================\n")
cat(" 步骤 10 补充: 尾部风险回测 (VaR & ES 1% 和 5%) 与 DQ 检验\n")
cat("========================================================================\n")

# 简单的动态分位数测试 (Dynamic Quantile Test / Cavaliere et al. 简版)
# 用于检查超额损失是否存在自相关
dq_test <- function(hit_sequence, df = 4) {
  # hit_sequence 是 0-1 序列，1表示突破 VaR
  T_len <- length(hit_sequence)
  p_target <- mean(hit_sequence)
  if (sum(hit_sequence) == 0 || sum(hit_sequence) == T_len) return(0.0) # 退化情况

  # 构造回归：Hit_t 对其自身的滞后项进行多元回归
  centered_hit <- hit_sequence - p_target
  lhs <- centered_hit[(df + 1):T_len]

  rhs <- matrix(NA, nrow = length(lhs), ncol = df)
  for (i in 1:df) {
    rhs[, i] <- centered_hit[(df + 1 - i):(T_len - i)]
  }

  fit_lm <- lm(lhs ~ rhs)
  r2 <- summary(fit_lm)$r.squared
  statistic <- T_len * r2
  p_val <- 1 - pchisq(statistic, df = df)
  return(p_val)
}

evaluate_tail_risk <- function(pred_var, alpha = 0.05) {
  # 根据方差计算各模型的 VaR。由于使用的是学生t分布(假定标准df=6作为通用评估，或从模型中提取)
  # 这里为了 benchmark 公平，统一采用标准化 t 分布分位数 (假设自由度为 6)
  df_assumed <- 6
  q_t <- qt(alpha, df = df_assumed) * sqrt((df_assumed - 2) / df_assumed)

  # VaR 预测
  pred_sigma <- sqrt(pred_var)
  VaR <- q_t * pred_sigma

  # 突破指标 (Hit)
  hits <- as.integer(y_test < VaR)
  actual_alpha <- mean(hits)

  # DQ 检验 P 值
  dq_p <- dq_test(hits, df = 3)

  # 预期尾部损失 (Expected Shortfall / ES) 经验计算
  es_loss <- mean(y_test[hits == 1])

  list(actual_alpha = actual_alpha, dq_p = dq_p, es_loss = es_loss)
}


cat("\n========================================================================\n")
cat(" 步骤 10.3 补充: 经济显著性 —— 波动率目标配置策略 (Annualized Target = 15%)\n")
cat("========================================================================\n")

run_volatility_targeting <- function(pred_var, target_annual_vol = 0.15) {
  # 假设周度数据 (52周/年)，如果是日度请改为 252
  freq <- 52
  target_weekly_vol <- target_annual_vol / sqrt(freq)

  # 预测的周度标准差
  pred_sigma <- sqrt(pred_var)

  # 资产权重 (风险资产分配权重, 允许杠杆则不截断，不允许则 pmin(weight, 1))
  # 论文标准：w_t = sigma_target / sigma_forecast
  weights <- target_weekly_vol / pred_sigma
  weights <- pmin(weights, 2.0) # 限制最大 2 倍杠杆防止极端值崩溃

  # 策略收益率 (假设无风险收益率为 0)
  portfolio_returns <- weights * y_test

  # 经济指标计算
  cum_returns <- cumprod(1 + portfolio_returns / 100) # y 是百分比，转化为小数
  ann_return <- mean(portfolio_returns) * freq
  ann_vol <- sd(portfolio_returns) * sqrt(freq)
  sharpe <- if(ann_vol > 0) ann_return / ann_vol else 0

  # 最大回撤 (Maximum Drawdown)
  cum_zoo <- zoo(cum_returns)
  max_dd <- max(1 - cum_zoo / cummax(cum_zoo), na.rm = TRUE)

  # 确定性等价收益率 (CER, 风险厌恶系数设为 3)
  gamma_risk <- 3
  cer <- ann_return - 0.5 * gamma_risk * (ann_vol^2)

  list(sharpe = sharpe, max_dd = max_dd, cer = cer, ann_return = ann_return)
}


cat("\n========================================================================\n")
cat(" 最终总主控输出汇总报告 (统计性能 + 尾部风险 + 经济效益)\n")
cat("========================================================================\n")

# 创建最终合并大表格
summary_metrics <- data.frame(
  Model = character(),
  QLIKE = numeric(),
  VaR_Hit_5pct = numeric(),
  DQ_P_value = numeric(),
  Sharpe_Ratio = numeric(),
  Max_Drawdown = numeric(),
  CER = numeric(),
  stringsAsFactors = FALSE
)

# 遍历收集到的所有模型结果进行统一结算
for (m_name in names(benchmark_results)) {
  res <- benchmark_results[[m_name]]

  # 提取测试集的预测方差
  if (is.null(res$forecasts$var_test)) next
  p_var <- res$forecasts$var_test

  # 1. QLIKE
  q_val <- qlike(actual_var_test, p_var)

  # 2. Tail Risk (5% VaR)
  tail_res <- evaluate_tail_risk(p_var, alpha = 0.05)

  # 3. 经济显著性
  econ_res <- run_volatility_targeting(p_var, target_annual_vol = 0.15)

  # 载入大表
  summary_metrics <- rbind(summary_metrics, data.frame(
    Model = m_name,
    QLIKE = round(q_val, 4),
    VaR_Hit_5pct = round(tail_res$actual_alpha, 4),
    DQ_P_value = round(tail_res$dq_p, 4),
    Sharpe_Ratio = round(econ_res$sharpe, 4),
    Max_Drawdown = round(econ_res$max_dd, 4),
    CER = round(econ_res$cer, 4)
  ))
}

print(summary_metrics)


cat("\n========================================================================\n")
cat(" 步骤 10.2 补充: 正式 Diebold-Mariano (DM) 显著性检验矩阵\n")
cat("========================================================================\n")

# 以列表中第一个模型（通常是 sGARCH 或某个经典模型）作为基准进行配对检验
if (nrow(summary_metrics) >= 2) {
  cat("正在进行相对于第一个模型的 DM 检验 (QLIKE 损失空间)...\n")
  base_model_name <- summary_metrics$Model[1]
  base_var <- benchmark_results[[base_model_name]]$forecasts$var_test
  loss_base <- actual_var_test / base_var - log(actual_var_test / base_var) - 1

  for (i in 2:nrow(summary_metrics)) {
    comp_model_name <- summary_metrics$Model[i]
    comp_var <- benchmark_results[[comp_model_name]]$forecasts$var_test
    loss_comp <- actual_var_test / comp_var - log(actual_var_test / comp_var) - 1

    # 运行经典 DM 检验 (单步预测 h=1)
    dm_out <- forecast::dm.test(loss_base, loss_comp, alternative = "greater", h = 1)
    cat(sprintf("  - %s 是否显著优于 %s? DM统计量: %.3f, P值: %.4f\n",
                comp_model_name, base_model_name, dm_out$statistic, dm_out$p.value))
  }
}

cat("\n所有补充评估指标计算完毕，报告生成成功。\n")
