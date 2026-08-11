library(fpp)
library(forecast)
library(gdata)
options(digits=16)

# Simple exponential smoothing
oildata <- window(oil,start=1996,end=2007)
oil_fpp1 <- ses(oildata, alpha=0.2, initial="simple", h=17)
oil_fpp2 <- ses(oildata, alpha=0.6, initial="simple", h=17)
oil_fpp3 <- ses(oildata, h=17)
oil_ets <- forecast(ets(oildata), h=17)

# Holt's linear trend method
air <- window(ausair,start=1990,end=2004)
air_fpp1 <- holt(air, alpha=0.8, beta=0.2, damped=FALSE, initial="simple", h=14)
air_fpp2 <- holt(air, alpha=0.8, beta=0.2, damped=TRUE, initial="optimal", h=14)
air_ets <- forecast(ets(air, 'AAN', damped=TRUE), h=14)

# Holt-Winters seasonal method
aust <- window(austourists,start=2005)
aust_fpp1 <- hw(aust, seasonal="additive", h=5)
aust_ets1 <- forecast(ets(aust, 'AAA', damped=FALSE), h=5)
aust_ets2 <- forecast(ets(aust, 'AAA', damped=TRUE), h=5)
aust_ets3 <- forecast(ets(aust, 'ANA'), h=5)

# Save params / output
params <- data.frame(
  oil_fpp1=c(oil_fpp1$model$par['alpha'], NA, NA, NA, NA, oil_fpp1$model$par['l'], NA, NA, NA, NA, oil_fpp1$model$sigma2, oil_fpp1$model$SSE, NA, NA),
  oil_fpp2=c(oil_fpp2$model$par['alpha'], NA, NA, NA, NA, oil_fpp2$model$par['l'], NA, NA, NA, NA, oil_fpp2$model$sigma2, oil_fpp2$model$SSE, NA, NA),
  oil_fpp3=c(oil_fpp3$model$par['alpha'], NA, NA, NA, NA, oil_fpp3$model$par['l'], NA, NA, NA, NA, oil_fpp3$model$sigma2, NA, oil_fpp3$model$loglik, oil_fpp3$model$mse),
  oil_ets=c(oil_ets$model$par['alpha'], NA, NA, NA, NA, oil_ets$model$par['l'], NA, NA, NA, NA, oil_ets$model$sigma2, NA, oil_ets$model$loglik, oil_ets$model$mse),
  air_fpp1=c(air_fpp1$model$par[1:2], NA, NA, NA, air_fpp1$model$par[3:4], NA, NA, NA, air_fpp1$model$sigma2, air_fpp1$model$SSE, NA, NA),
  air_fpp2=c(air_fpp2$model$par[4], NA, air_fpp2$model$par[5], NA, air_fpp2$model$par[1:3], NA, NA, NA, air_fpp2$model$sigma2, NA, air_fpp2$model$loglik, air_fpp2$model$mse),
  air_ets=c(air_ets$model$par[1], NA, air_ets$model$par[2], NA, air_ets$model$par[3:5], NA, NA, NA, air_ets$model$sigma2, NA, air_ets$model$loglik, air_ets$model$mse),
  aust_fpp1=c(aust_fpp1$model$par[1], NA, aust_fpp1$model$par[2:3], NA, aust_fpp1$model$par[4:8], aust_fpp1$model$sigma2, NA, aust_fpp1$model$loglik, aust_fpp1$model$mse),
  aust_ets1=c(aust_ets1$model$par[1], NA, aust_ets1$model$par[2:3], NA, aust_ets1$model$par[4:8], aust_ets1$model$sigma2, NA, aust_ets1$model$loglik, aust_ets1$model$mse),
  aust_ets2=c(aust_ets2$model$par[1], NA, aust_ets2$model$par[2:9], aust_ets2$model$sigma2, NA, aust_ets2$model$loglik, aust_ets2$model$mse),
  aust_ets3=c(aust_ets3$model$par[1], NA, NA, aust_ets3$model$par[2], NA, aust_ets3$model$par[3], NA, aust_ets3$model$par[4:6], aust_ets3$model$sigma2, NA, aust_ets3$model$loglik, aust_ets3$model$mse))
row.names(params) <- c('alpha', 'beta_star', 'beta', 'gamma', 'phi', 'l0', 'b0', 's0_0', 's0_1', 's0_2', 'sigma2', 'sse', 'llf', 'mse')

# Save predict / forecast
predict <- data.frame(
  oil_fpp1_mean=c(oil_fpp1$fitted, oil_fpp1$mean),
  oil_fpp1_lower=c(rep(NA, 12), oil_fpp1$lower[,'95%']),
  oil_fpp1_upper=c(rep(NA, 12), oil_fpp1$upper[,'95%']),

  oil_fpp2_mean=c(oil_fpp2$fitted, oil_fpp2$mean),
  oil_fpp2_lower=c(rep(NA, 12), oil_fpp2$lower[,'95%']),
  oil_fpp2_upper=c(rep(NA, 12), oil_fpp2$upper[,'95%']),

  oil_fpp3_mean=c(oil_fpp3$fitted, oil_fpp3$mean),
  oil_fpp3_lower=c(rep(NA, 12), oil_fpp3$lower[,'95%']),
  oil_fpp3_upper=c(rep(NA, 12), oil_fpp3$upper[,'95%']),
  
  oil_ets_mean=c(oil_ets$fitted, oil_ets$mean),
  oil_ets_lower=c(rep(NA, 12), oil_ets$lower[,'95%']),
  oil_ets_upper=c(rep(NA, 12), oil_ets$upper[,'95%']),

  air_fpp1_mean=c(air_fpp1$fitted, air_fpp1$mean),
  air_fpp1_lower=c(rep(NA, 15), air_fpp1$lower[,'95%']),
  air_fpp1_upper=c(rep(NA, 15), air_fpp1$upper[,'95%']),

  air_fpp2_mean=c(air_fpp2$fitted, air_fpp2$mean),
  air_fpp2_lower=c(rep(NA, 15), air_fpp2$lower[,'95%']),
  air_fpp2_upper=c(rep(NA, 15), air_fpp2$upper[,'95%']),
  
  air_ets_mean=c(air_ets$fitted, air_ets$mean),
  air_ets_lower=c(rep(NA, 15), air_ets$lower[,'95%']),
  air_ets_upper=c(rep(NA, 15), air_ets$upper[,'95%']),
  
  aust_fpp1_mean=c(aust_fpp1$fitted, aust_fpp1$mean),
  aust_fpp1_lower=c(rep(NA, 24), aust_fpp1$lower[,'95%']),
  aust_fpp1_upper=c(rep(NA, 24), aust_fpp1$upper[,'95%']),
  
  aust_ets1_mean=c(aust_ets1$fitted, aust_ets1$mean),
  aust_ets1_lower=c(rep(NA, 24), aust_ets1$lower[,'95%']),
  aust_ets1_upper=c(rep(NA, 24), aust_ets1$upper[,'95%']),
  
  aust_ets2_mean=c(aust_ets2$fitted, aust_ets2$mean),
  aust_ets2_lower=c(rep(NA, 24), aust_ets2$lower[,'95%']),
  aust_ets2_upper=c(rep(NA, 24), aust_ets2$upper[,'95%']),
  
  aust_ets3_mean=c(aust_ets3$fitted, aust_ets3$mean),
  aust_ets3_lower=c(rep(NA, 24), aust_ets3$lower[,'95%']),
  aust_ets3_upper=c(rep(NA, 24), aust_ets3$upper[,'95%']))

# Save estimated states
oil_fpp1_states <- data.frame(oil_fpp1$model$states)
colnames(oil_fpp1_states) <- paste("oil_fpp1", colnames(oil_fpp1_states), sep="_")

oil_fpp2_states <- data.frame(oil_fpp2$model$states)
colnames(oil_fpp2_states) <- paste("oil_fpp2", colnames(oil_fpp2_states), sep="_")

oil_fpp3_states <- data.frame(oil_fpp3$model$states)
colnames(oil_fpp3_states) <- paste("oil_fpp3", colnames(oil_fpp3_states), sep="_")

oil_ets_states <- data.frame(oil_ets$model$states)
colnames(oil_ets_states) <- paste("oil_ets", colnames(oil_ets_states), sep="_")

air_fpp1_states <- data.frame(air_fpp1$model$states)
colnames(air_fpp1_states) <- paste("air_fpp1", colnames(air_fpp1_states), sep="_")

air_fpp2_states <- data.frame(air_fpp2$model$states)
colnames(air_fpp2_states) <- paste("air_fpp2", colnames(air_fpp2_states), sep="_")

air_ets_states <- data.frame(air_ets$model$states)
colnames(air_ets_states) <- paste("air_ets", colnames(air_ets_states), sep="_")

aust_fpp1_states <- data.frame(aust_fpp1$model$states)
colnames(aust_fpp1_states) <- paste("aust_fpp1", colnames(aust_fpp1_states), sep="_")

aust_ets1_states <- data.frame(aust_ets1$model$states)
colnames(aust_ets1_states) <- paste("aust_ets1", colnames(aust_ets1_states), sep="_")

aust_ets2_states <- data.frame(aust_ets2$model$states)
colnames(aust_ets2_states) <- paste("aust_ets2", colnames(aust_ets2_states), sep="_")

aust_ets3_states <- data.frame(aust_ets3$model$states)
colnames(aust_ets3_states) <- paste("aust_ets3", colnames(aust_ets3_states), sep="_")

states <- cbindX(
  oil_fpp1_states, oil_fpp2_states, oil_fpp3_states, oil_ets_states,
  air_fpp1_states, air_fpp2_states, air_ets_states,
  aust_fpp1_states, aust_ets1_states, aust_ets2_states, aust_ets3_states)

write.csv(params, 'exponential_smoothing_params.csv')
write.csv(predict, 'exponential_smoothing_predict.csv')
write.csv(states, 'exponential_smoothing_states.csv')
