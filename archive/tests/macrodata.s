dta <- read.csv('../../datasets/macrodata/macrodata.csv', header = TRUE)
attach(dta)
library(systemfit)

demand <- realcons + realinv + realgovt
c.1 <- realcons[-203]
y.1 <- demand[-203]
yd <- demand[-1] - y.1
eqConsump <- realcons[-1] ~ demand[-1] + c.1
eqInvest <- realinv[-1] ~ tbilrate[-1] + yd
system <- list( Consumption = eqConsump, Investment = eqInvest)
instruments <- ~ realgovt[-1] + tbilrate[-1] + c.1 + y.1
# 2SLS
greene2sls <- systemfit( system, "2SLS", inst = instruments, methodResidCov = "noDfCor" )
print(summary(greene2sls))

greene3sls <- systemfit( system, "3SLS", inst = instruments, methodResidCov = "noDfCor" )
print(summary(greene3sls))


# Python code for finding the dynamics
#
# Could have done this in R
#
#gamma = np.array([[1,0,1],[0,1,1],[-.058438620413,-16.5359646223,1]])
#phi = np.array([[-.99200661799,0,0],[0,0,0],[0,-16.5359646223,0]])
#Delta = np.dot(-phi,np.linalg.inv(gamma))
#delta = np.zeros((2,2))
#delta[0,0]=Delta[0,0]
#delta[0,1]=Delta[0,-1]
#delta[1,0]=Delta[-1,0]
#delta[1,1]=Delta[-1,-1]
#np.eigvals(delta)
#np.max(_)

