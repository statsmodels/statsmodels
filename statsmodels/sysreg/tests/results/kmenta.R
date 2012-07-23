library("systemfit")
data( "Kmenta" )
eqDemand <- consump ~ price + income
eqSupply <- consump ~ price + farmPrice + trend
inst <- ~ income + farmPrice + trend
system <- list( demand = eqDemand, supply = eqSupply )

fit2sls <- systemfit( system, method = "2SLS", inst = inst, data = Kmenta,
methodResidCov='noDfCor')
fitw2sls <- systemfit( system, method = "W2SLS", inst = inst, data = Kmenta,
methodResidCov='noDfCor')
fit3sls <- systemfit( system, method = "3SLS", inst = inst, data = Kmenta,
methodResidCov='noDfCor')
# The methods for calculating the 3SLS estimator lead to identical results if 
# the same instruments are used in all equations.

fitols <- systemfit( system, method = "OLS", data = Kmenta,
methodResidCov='noDfCor')
fitsur <- systemfit( system, method = "SUR", data = Kmenta,
methodResidCov='noDfCor', maxiter=1)

