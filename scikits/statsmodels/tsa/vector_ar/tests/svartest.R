library("vars")

#data <- read.csv("/home/skipper/statsmodels/statsmodels-skipper/scikits/statsmodels/datasets/macrodata/macrodata.csv")
#data <- read.csv("/home/bart/statsmodels/scikits/statsmodels/datasets/macrodata/macrodata.csv")
data <- read.csv("C:\\statsmodels\\statsmodels-bartbkr\\scikits\\statsmodels\\datasets\\macrodata\\macrodata.csv")
names <- colnames(data)
data <- log(data[c('realgdp','realcons','realinv')])
data <- sapply(data, diff)
data = ts(data, start=c(1959,2), frequency=4)

var <-VAR(data, p=3, type= "const")
amat <- matrix(0,3,3)
amat[1,1] <- 1
amat[2,1] <- NA
amat[3,1] <- NA
amat[2,2] <- 1
amat[3,2] <- NA
amat[3,3] <- 1
bmat <- diag(3)
diag(bmat) <- NA
svar <- SVAR(var, estmethod = 'scoring', Bmat=bmat, Amat=amat)
plot(irf(svar, n.ahead=30, impulse = 'realgdp'))
#myirf <- plot(irf(myvar, impulse = "realgdp", response = c("realgdp", "realcons", "realinv"), boot=TRUE, n.ahead=30, ci=0.95))
#plot.irf()
