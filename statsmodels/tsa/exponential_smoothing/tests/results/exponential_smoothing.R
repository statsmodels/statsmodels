library(fpp)
options(digits=16)

# Simple exponential smoothing
oildata <- window(oil,start=1996,end=2007)
oil_fit1 <- ses(oildata, alpha=0.2, initial="simple", h=17)
oil_fit2 <- ses(oildata, alpha=0.6, initial="simple", h=17)
oil_fit3 <- ses(oildata, h=17)

# Holt's linear trend method
air <- window(ausair,start=1990,end=2004)
air_fit1 <- holt(air, alpha=0.8, beta=0.2, initial="simple", h=14)
air_fit2 <- holt(air, alpha=0.8, beta=0.2, initial="simple", exponential=TRUE, h=14)
air_fit3 <- holt(air, alpha=0.8, beta=0.2, damped=TRUE, initial="simple", h=14)

# Holt-Winters seasonal method
aust <- window(austourists,start=2005)
aust_fit1 <- hw(aust, seasonal="additive", h=5)
aust_fit2 <- hw(aust, seasonal="multiplicative", h=5)

# Save fitted and forecasts
fitted <- data.frame(
  oil_fit1=c(oil_fit1$fitted, oil_fit1$mean),
  oil_fit2=c(oil_fit2$fitted, oil_fit2$mean),
  oil_fit3=c(oil_fit3$fitted, oil_fit3$mean),
  air_fit1=c(air_fit1$fitted, air_fit1$mean),
  air_fit2=c(air_fit2$fitted, air_fit2$mean),
  air_fit3=c(air_fit3$fitted, air_fit3$mean),
  aust_fit1=c(aust_fit1$fitted, aust_fit1$mean),
  aust_fit2=c(aust_fit2$fitted, aust_fit2$mean))

write.csv(fitted, 'fitted.csv')
