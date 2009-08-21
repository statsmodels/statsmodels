d <- read.table('./sunspots.csv', sep=',', header=T)
attach(d)

m1 <- ar(SUNACTIVITY, aic=FALSE, order.max=4)
