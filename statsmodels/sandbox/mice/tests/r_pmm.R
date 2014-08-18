N<-250;
x1<-rbinom(N,1,prob=.4)  #draw from a binomial dist with probability=.4
x2<-rnorm(N,0,1)         #draw from a normal dist with mean=0, sd=1
x3<-rnorm(N,-10,1)
y<--1+1*x1-1*x2+1*x3+rnorm(N,0,1)  #simulate linear regression data with a normal error (sd=1)

#Generate MAR data

alpha.1<-exp(16+2*y-x2)/(1+exp(16+2*y-x2));
alpha.2<-exp(3.5+.7*y)/(1+exp(3.5+.7*y));
alpha.3<-exp(-13-1.2*y-x1)/(1+exp(-13-1.2*y-x1));


r.x1.mar<-rbinom(N,1,prob=alpha.1)
r.x2.mar<-rbinom(N,1,prob=alpha.2)
r.x3.mar<-rbinom(N,1,prob=alpha.3)
x1.mar<-x1*(1-r.x1.mar)+r.x1.mar*99999  #x1.mar=x1 if not missing, 99999 if missing
x2.mar<-x2*(1-r.x2.mar)+r.x2.mar*99999
x3.mar<-x3*(1-r.x3.mar)+r.x3.mar*99999
x1.mar[x1.mar==99999]=NA                  #change 99999 to NA (R's notation for missing)
x2.mar[x2.mar==99999]=NA
x3.mar[x3.mar==99999]=NA

require(mice)
data = as.data.frame(cbind(x1.mar,x2.mar,x3.mar))
data$x1.mar = as.factor(data$x1.mar)
nrep = 500
params = array(0, nrep)
# for (i in 1:nrep):
# {
  imp_pmm = mice(data, method="pmm", maxit=50)
  pooled = pool(with(imp_pmm,glm(x1.mar~x2.mar+x3.mar,family=binomial)))
  summary(pooled)
  
# }

setwd("C:/Users/Frank/Dropbox/statsmodels/statsmodels/sandbox/mice/tests")

write.csv(cbind(pooled$u[1:5], pooled$u[21:25], pooled$u[41:45]), "cov.csv", row.names=FALSE)
write.csv(pooled$qhat, "params.csv", row.names=FALSE)
write.csv(data, "missingdata.csv", row.names=FALSE)