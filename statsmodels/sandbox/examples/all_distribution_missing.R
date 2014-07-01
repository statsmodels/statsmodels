  setwd("C:/Users/Frank/Documents/GitHub/statsmodels/statsmodels/sandbox/examples")
  size = 1000
  cor = 0.9
  mu = rep(0,3)
  sig = matrix(cor, nrow=3, ncol=3) + diag(3) * (1-cor)
  draws = rmvnorm(n=size, mean=mu, sigma=sig)
  unidraws = pnorm(draws)
  normdraws = draws[,1]
  berndraws = qbinom(1-unidraws[,2], 1, 0.75)
  poisdraws = qpois(unidraws[,3], 5)
  y<--1+1*berndraws-1*normdraws+1*poisdraws+rnorm(size,0,1) 
  
  alpha.1<-exp(-16+2*y-normdraws)/(1+exp(-16+2*y-normdraws));
  alpha.2<-exp(-3.5+.7*y)/(1+exp(-3.5+.7*y));
  alpha.3<-exp(-6+1.2*y-berndraws)/(1+exp(-6+1.2*y-berndraws));
  
  
  r.berndraws.mar<-rbinom(N,1,prob=alpha.1)
  r.normdraws.mar<-rbinom(N,1,prob=alpha.2)
  r.poisdraws.mar<-rbinom(N,1,prob=alpha.3)
  berndraws.mar<-berndraws*(1-r.berndraws.mar)+r.berndraws.mar*99999  #berndraws.mar=berndraws if not missing, 99999 if missing
  normdraws.mar<-normdraws*(1-r.normdraws.mar)+r.normdraws.mar*99999
  poisdraws.mar<-poisdraws*(1-r.poisdraws.mar)+r.poisdraws.mar*99999
  berndraws.mar[berndraws.mar==99999]=NA                  #change 99999 to NA (R's notation for missing)
  normdraws.mar[normdraws.mar==99999]=NA
  poisdraws.mar[poisdraws.mar==99999]=NA
  
  write.csv(cbind(berndraws.mar,normdraws.mar,poisdraws.mar), "missingfull.csv", row.names=FALSE)
  
  require(mice)
  
  data = read.csv("missingfull.csv")
  
  imp = mice(data, m=20, method="pmm", maxit=20)
  
  mod = glm(berndraws.mar ~ normdraws.mar + poisdraws.mar, data, family=binomial)
  
  pooled = pool(with(imp, mod))
  print(summary(pooled))