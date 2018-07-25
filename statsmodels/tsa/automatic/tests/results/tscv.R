library("forecast")
macrodata <- read.csv('../../../../datasets/macrodata/macrodata.csv')
y <- ts(macrodata$infl)
far2 <- function(x, h){forecast(Arima(x, order=c(2,0,2)), h=h)}
e <- tsCV(y, far2, h=1, window=40)
