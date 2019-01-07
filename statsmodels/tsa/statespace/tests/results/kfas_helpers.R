# Helper functions for working with KFAS objects

cbind.fill <- function(...){
  nm <- list(...) 
  nm <- lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow)) 
  do.call(cbind, lapply(nm, function (x) 
    rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}

flatten <- function(m) {
  f <- t(apply(m, 3, c))
  dm <- dim(m)
  df <- dim(f)
  if(df[1] == 1 && dm[3] != 1) {
    f <- t(f)
  }
  f
}

kfas.output <- function(kf) {
  # Dimensions
  n = nrow(kf$model$y)
  p = ncol(kf$model$y)
  p2 = p^2
  m = nrow(kf$model$T)
  m2 = m^2
  mp = m * p
  r = nrow(kf$model$Q)
  r2 = r^2
  
  # Construct the output dataframe
  # r = scaled_smoothed_estimator
  # N = scaled_smoothed_estimator_cov
  # m = forecasts
  # v = forecasts_error
  # F = forecasts_error_cov
  # a = predicted_state
  # P = predicted_state_cov
  # mu = filtered_forecasts
  # alphahat = smoothed_state
  # V = smoothed_state_cov
  # muhat = smoothed_forecasts
  # etahat = smoothed_state_disturbance
  # V_eta = smoothed_state_disturbance_cov
  # epshat = smoothed_measurement_disturbance
  # V_eps = smoothed_measurement_disturbance_cov
  out <- as.data.frame(with(kf, cbind.fill(
    t(r), t(r0), t(r1),
    flatten(N), flatten(N0), flatten(N1), flatten(N2),
    m, v, t(F), t(Finf),
    flatten(K), flatten(Kinf),
    a, flatten(P), flatten(Pinf),
    att, flatten(Ptt),
    alphahat, flatten(V),
    muhat, flatten(V_mu),
    etahat, flatten(V_eta),
    epshat, t(V_eps),
    logLik
  )))
  names(out) <- c(
    paste('r', 1:m, sep='_'), paste('r0', 1:m, sep='_'), paste('r1', 1:m, sep='_'),
    paste('N', 1:m2, sep='_'), paste('N0', 1:m2, sep='_'), paste('N1', 1:m2, sep='_'), paste('N2', 1:m2, sep='_'),
    paste('m', 1:p, sep='_'), paste('v', 1:p, sep='_'), paste('F', 1:p, sep='_'), paste('Finf', 1:p, sep='_'),
    paste('K', 1:mp, sep='_'), paste('Kinf', 1:mp, sep='_'),
    paste('a', 1:m, sep='_'), paste('P', 1:m2, sep='_'), paste('Pinf', 1:m2, sep='_'),
    paste('att', 1:m, sep='_'), paste('Ptt', 1:m2, sep='_'),
    paste('alphahat', 1:m, sep='_'), paste('V', 1:m2, sep='_'),
    paste('muhat', 1:p, sep='_'), paste('Vmu', 1:p2, sep='_'),
    paste('etahat', 1:r, sep='_'), paste('Veta', 1:r2, sep='_'),
    paste('epshat', 1:p, sep='_'), paste('Veps', 1:p, sep='_'),
    'llf'
  )
  return(out)
}


kfas.llf <- function(kf) {
  # Dimensions
  n = nrow(kf$model$y)
  p = ncol(kf$model$y)
  m = nrow(kf$model$T)
  r = nrow(kf$model$Q)
  d = kf$d
  F = kf$F
  Finf = kf$Finf
  v = kf$v
  
  -((n * p / 2) * log(pi * 2) +
      0.5 * sum(log(Finf[1:d])) +
      0.5 * sum(log(F[(d+1):10]) + (v^2 / t(F))[(d+1):10]))
}
