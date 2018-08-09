library("forecast")
macrodata <- read.csv('../../../../datasets/macrodata/macrodata.csv')
y1 <- ts(macrodata$cpi)
fit1 <- auto.arima(y1)

y2 <- ts(macrodata$infl)
fit2 <- auto.arima(y2)

data_new <- read.csv('../CPIAPPNS.csv')
y3<-ts(data1$CPIAPPNS)
fit3 <- auto.arima(y3)
