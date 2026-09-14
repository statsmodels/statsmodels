library("forecast")
macrodata <- read.csv('../../../../datasets/macrodata/macrodata.csv')
y <- ts(macrodata$cpi)
fit <- ets(y,'AZZ')

y <- ts(macrodata$infl)
fit <- ets(y,'AZZ')

data_new <- read.csv('../CPIAPPNS.csv')
y<-ts(data_new$CPIAPPNS)
fit3 <- ets(y,'AZZ')
