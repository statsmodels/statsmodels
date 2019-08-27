library(KFAS)
# library(rucm)
options(digits=10)

dta <- read.csv('datasets/macrodata/macrodata.csv')

# Irregular (ntrend)
mod_ntrend <- SSModel(
  dta$unemp ~ -1 + SSMcustom(Z=matrix(0), matrix(0), matrix(0),
                             matrix(0)),
  H=matrix(NA))
res_ntrend <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_ntrend,
                     method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_ntrend$optim.out$par)) # 36.74687342
print(res_ntrend$optim.out$value)    # 653.8562525

# ----------------------------------------------------------------------------

# Irregular + Deterministic trend (dconstant)
mod_dconstant <- SSModel(dta$unemp ~ SSMtrend(Q=matrix(0)), H=matrix(NA))
res_dconstant <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_dconstant,
                        method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_dconstant$optim.out$par)) # 2.127438969
print(res_dconstant$optim.out$value)    # 365.5289923

# ----------------------------------------------------------------------------

# Local level (llevel)
mod_llevel <- SSModel(dta$unemp ~ SSMtrend(Q=matrix(NA)), H=matrix(NA))
res_llevel <- fitSSM(inits=c(log(var(dta$unemp)), log(var(dta$unemp))),
                     model=mod_llevel, method="BFGS",
                     control=list(REPORT=1, trace=1))

print(exp(res_llevel$optim.out$par)) # [1.182078808e-01, 4.256647886e-06]
print(res_llevel$optim.out$value)    # 70.97242557

# ----------------------------------------------------------------------------

# Random walk (rwalk)
mod_rwalk <- SSModel(dta$unemp ~ SSMtrend(Q=matrix(NA)), H=matrix(0))
res_rwalk <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_rwalk,
                    method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_rwalk$optim.out$par)) # 0.1182174646
print(res_rwalk$optim.out$value)    # 70.96771641

# ----------------------------------------------------------------------------

# Deterministic trend (dtrend)
mod_dtrend <- SSModel(dta$unemp ~ SSMtrend(2, Q=list(matrix(0), matrix(0))),
                      H=matrix(NA))
res_dtrend <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_dtrend,
                     method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_dtrend$optim.out$par)) # 2.134137554
print(res_dtrend$optim.out$value)    # 370.7758666

# ----------------------------------------------------------------------------

# Local level with deterministic trend (lldtrend)
mod_lldtrend <- SSModel(dta$unemp ~ SSMtrend(2, Q=list(matrix(NA), matrix(0))),
                        H=matrix(NA))
res_lldtrend <- fitSSM(inits=c(log(var(dta$unemp)), log(var(dta$unemp))),
                       model=mod_lldtrend, method="BFGS",
                       control=list(REPORT=1, trace=1))

print(exp(res_lldtrend$optim.out$par)) # [1.184455029e-01, 4.457592057e-06]
print(res_lldtrend$optim.out$value)    # 73.47291031

# ----------------------------------------------------------------------------

# Random walk with drift (rwdrift)
mod_rwdrift <- SSModel(dta$unemp ~ SSMtrend(2, Q=list(matrix(NA), matrix(0))),
                       H=matrix(0))
res_rwdrift <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_rwdrift,
                      method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_rwdrift$optim.out$par)) # [0.1184499547]
print(res_rwdrift$optim.out$value)    # 73.46798576

# ----------------------------------------------------------------------------

# Local linear trend (lltrend)
mod_lltrend <- SSModel(dta$unemp ~ SSMtrend(2, Q=list(matrix(NA), matrix(NA))),
                       H=matrix(NA))
res_lltrend <- fitSSM(inits=c(log(var(dta$unemp)), log(var(dta$unemp)),
                              log(var(dta$unemp))),
                      model=mod_lltrend, method="BFGS",
                      control=list(REPORT=1, trace=1))

print(exp(res_lltrend$optim.out$par)) # [1.008704925e-02, 6.091760810e-02,
                                      #  1.339852549e-06]
print(res_lltrend$optim.out$value)    # 31.15640107

# ----------------------------------------------------------------------------

# Smooth trend (strend)
mod_strend <- SSModel(dta$unemp ~ SSMtrend(2, Q=list(matrix(0), matrix(NA))),
                      H=matrix(NA))
res_strend <- fitSSM(inits=c(log(var(dta$unemp)), log(var(dta$unemp))),
                     model=mod_strend, method="BFGS",
                     control=list(REPORT=1, trace=1))

print(exp(res_strend$optim.out$par)) # [0.0753064234342, 0.0008824099119]
print(res_strend$optim.out$value)    # 31.92261408

# ----------------------------------------------------------------------------

# Random trend (rtrend)
mod_rtrend <- SSModel(dta$unemp ~ SSMtrend(2, Q=list(matrix(0), matrix(NA))),
                      H=matrix(0))
res_rtrend <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_rtrend,
                     method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_rtrend$optim.out$par)) # [0.08054724989]
print(res_rtrend$optim.out$value)    # 32.05607557

# ----------------------------------------------------------------------------

# Cycle (exact diffuse) (with fixed period=10, cycle variance=0.1)
mod_cycle <- SSModel(dta$unemp ~ -1 + SSMcycle(10, Q=matrix(0.1)),
                     H=matrix(NA))
res_cycle <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_cycle,
                    method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_cycle$optim.out$par)) # [37.57197079]
print(res_cycle$optim.out$value)    # 656.6568684

# Cycle (approximate diffuse) (with fixed period=10, cycle variance=0.1)
# Note: matching this requires setting loglikelihood_burn = 0
mod_cycle_approx_diffuse <- SSModel(
  dta$unemp ~ -1 + SSMcycle(10, Q=matrix(0.1), P1=diag(2)*1e6), H=matrix(NA))
res_cycle_approx_diffuse <- fitSSM(inits=c(log(var(dta$unemp))),
                     model=mod_cycle_approx_diffuse, method="BFGS",
                     control=list(REPORT=1, trace=1))

print(exp(res_cycle_approx_diffuse$optim.out$par)) # [37.57197224]
print(res_cycle_approx_diffuse$optim.out$value)    # 672.3102588

# ----------------------------------------------------------------------------

# Seasonal (exact diffuse) (with fixed period=4, seasonal variance=0.1)
mod_seasonal <- SSModel(
  dta$unemp ~ -1 + SSMseasonal(4, Q=matrix(0.1), sea.type='dummy'),
  H=matrix(NA))
res_seasonal <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_seasonal,
                       method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_seasonal$optim.out$par)) # [38.17043009]
print(res_seasonal$optim.out$value)    # 655.3337155

# Seasonal (approximate diffuse) (with fixed period=4, seasonal variance=0.1)
# Note: matching this requires setting loglikelihood_burn=0
mod_seasonal_approx_diffuse <- SSModel(
  dta$unemp ~ -1 + SSMseasonal(4, Q=matrix(0.1), sea.type='dummy',
                               P1=diag(rep(1e6, 3))),
  H=matrix(NA))
res_seasonal_approx_diffuse <- fitSSM(
  inits=c(log(var(dta$unemp))), model=mod_seasonal_approx_diffuse,
  method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_seasonal_approx_diffuse$optim.out$par)) # [38.1704278]
print(res_seasonal_approx_diffuse$optim.out$value)    # 678.8138005

# ----------------------------------------------------------------------------

# Trig Seasonal (exact diffuse)
# (with fixed period=5, full number of harmonics, seasonal variance=.05)
mod_trig_seasonal <- SSModel(
  dta$unemp ~ -1 + SSMseasonal(5, Q=0.05, sea.type='trigonometric'),
  H=matrix(NA))
res_trig_seasonal <- fitSSM(
  inits=c(log(var(dta$unemp))), model=mod_trig_seasonal, method="BFGS",
  control=list(REPORT=1, trace=1))

print(exp(res_trig_seasonal$optim.out$par)) # [38.95353592]
print(res_trig_seasonal$optim.out$value)    # 657.6629457

# Trig Seasonal (approximate diffuse)
# (with fixed period=5, full number of harmonics, seasonal variance=.05)
# Note: matching this requires setting loglikelihood_burn=0
mod_trig_seasonal_approx_diffuse <- SSModel(
  dta$unemp ~ -1 + SSMseasonal(5, Q=0.05, sea.type='trigonometric',
                               P1=diag(rep(1e6, 4))),
  H=matrix(NA))
res_trig_seasonal_approx_diffuse <- fitSSM(
  inits=c(log(var(dta$unemp))), model=mod_trig_seasonal_approx_diffuse,
  method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_trig_seasonal_approx_diffuse$optim.out$par)) # [38.95352534]
print(res_trig_seasonal_approx_diffuse$optim.out$value)    # 688.9697249

# ----------------------------------------------------------------------------

# Regression (exact diffuse)
# Note: matching this requires setting loglikelihood_burn = 0
mod_reg <- SSModel(dta$unemp ~ -1 + SSMregression(~-1+log(realgdp), data=dta),
                   H=matrix(NA))
res_reg <- fitSSM(inits=c(log(var(dta$unemp))), model=mod_reg,
                       method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_reg$optim.out$par)) # [2.215438082]
print(res_reg$optim.out$value)    # 371.7966543

# Regression (approximate diffuse)
# Note: matching this requires setting loglikelihood_burn = 0
mod_reg_approx_diffuse <- SSModel(
  dta$unemp ~ -1 + SSMregression(~-1+log(realgdp), data=dta, P1=diag(1)*1e6),
  H=matrix(NA))
res_reg_approx_diffuse <- fitSSM(
  inits=c(log(var(dta$unemp))), model=mod_reg_approx_diffuse,
  method="BFGS", control=list(REPORT=1, trace=1))

print(exp(res_reg_approx_diffuse$optim.out$par)) # [2.215447924]
print(res_reg_approx_diffuse$optim.out$value)    # 379.6233483

# ----------------------------------------------------------------------------

# Random trend + AR(1)
# Note: KFAS does not want to estimate these parameters, so just fix them
# to the MLE estimates from statsmodels and compare the loglikelihood
# mod.update([])
mod_rtrend_ar1 <- SSModel(
  dta$unemp ~ SSMtrend(2, Q=list(matrix(0), matrix(0.0609)))
    + SSMarima(ar=c(0.9592), Q=matrix(0.0097)),
  H=matrix(0))
out_rtrend_ar1 <- KFS(mod_rtrend_ar1)

print(out_rtrend_ar1$logLik)    # -31.15629379

# ----------------------------------------------------------------------------

# Local linear trend + Cycle + Seasonal + Regression + AR(1)
# (exact diffuse)
# mod.update([0.0001, 0.01, 0.06, 0.0001, 0.0001, 0.1, 2*np.pi / 10, 0.2])
mod_lltrend_cycle_seasonal_reg_ar1 <- SSModel(
  dta$unemp ~ SSMtrend(2, Q=list(matrix(0.01), matrix(0.06))) +
              SSMcycle(10, Q=matrix(0.0001)) +
              SSMseasonal(4, Q=matrix(0.0001)) +
              SSMregression(~-1+log(realgdp), data=dta) +
              SSMarima(ar=c(0.2), Q=matrix(0.1)),
H=matrix(0.0001))
out_lltrend_cycle_seasonal_reg_ar1 <- KFS(mod_lltrend_cycle_seasonal_reg_ar1)

print(out_lltrend_cycle_seasonal_reg_ar1$logLik)    # -105.8936568

# Local linear trend + Cycle + Seasonal + Regression + AR(1)
# (approximate diffuse)
# Note: matching this requires setting loglikelihood_burn = 0, and
# mod.update([0.0001, 0.01, 0.06, 0.0001, 0.0001, 0.1, 2*np.pi / 10, 0.2])
mod_lltrend_cycle_seasonal_reg_ar1_approx_diffuse <- SSModel(
  dta$unemp ~ SSMtrend(2, Q=list(matrix(0.01), matrix(0.06)), P1=diag(2)*1e6)
              + SSMcycle(10, Q=matrix(0.0001), P1=diag(2)*1e6)
              + SSMseasonal(4, Q=matrix(0.0001), P1=diag(3)*1e6)
              + SSMregression(~-1+log(realgdp), data=dta, P1=diag(1)*1e6)
              + SSMarima(ar=c(0.2), Q=matrix(0.1)),
  H=matrix(0.0001))
out_lltrend_cycle_seasonal_reg_ar1_approx_diffuse <- (
  KFS(mod_lltrend_cycle_seasonal_reg_ar1_approx_diffuse))

print(out_lltrend_cycle_seasonal_reg_ar1_approx_diffuse$logLik)  # -168.5258709
