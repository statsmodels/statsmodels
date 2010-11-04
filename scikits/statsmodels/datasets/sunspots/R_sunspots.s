d <- read.table('./sunspots.csv', sep=',', header=T)
attach(d)

mod_ols <- ar(SUNACTIVITY, aic=FALSE, order.max=9, method="ols", intercept=FALSE)
mod_yw <- ar(SUNACTIVITY, aic=FALSE, order.max=9, method="yw")
mod_burg <- ar(SUNACTIVITY, aic=FALSE, order.max=9, method="burg")
mod_mle <- ar(SUNACTIVITY, aic=FALSE, order.max=9, method="mle")

select_ols <- ar(SUNACTIVITY, aic=TRUE, method="ols")
