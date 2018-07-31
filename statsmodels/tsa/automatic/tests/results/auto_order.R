library("forecast")
macrodata <- read.csv('../../../../datasets/macrodata/macrodata.csv')
y1 <- ts(macrodata$cpi)
fit1 <- auto.arima(y1)

y2 <- ts(macrodata$infl)
fit2 <- auto.arima(y2)
