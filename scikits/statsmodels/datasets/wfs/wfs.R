# example of doing Poisson GLM with offset

dta <- read.csv('wfs.csv')
attach(dta)
y <- round(nwomen * nchild)
dur <- factor(dur)
res <- factor(res)
edu <- factor(edu)

fit.offset <- glm(y ~ dur + res + edu, offset = log(nwomen), family="poisson")
fit.nooffset <- glm(y ~ dur + res + edu, family="poisson")
