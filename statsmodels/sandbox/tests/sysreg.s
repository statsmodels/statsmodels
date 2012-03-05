# from the systemfit docs and sem docs
# depends systemfit and its dependencies
# depends sem
# depends on plm
# depends on R >= 2.9.0 (working on 2.9.2 but not on 2.8.1 at least)

library( systemfit )
data( "Kmenta" )
eqDemand <- consump ~ price + income
eqSupply <- consump ~ price + farmPrice + trend
system <- list( demand = eqDemand, supply = eqSupply )

## performs OLS on each of the equations in the system
fitols <- systemfit( system, data = Kmenta )

# all coefficients
coef( fitols )
coef( summary ( fitols ) )

modReg <- matrix(0,7,6)
colnames( modReg ) <- c( "demIntercept", "demPrice", "demIncome",
    "supIntercept", "supPrice2", "supTrend" )

# a lot of typing for a model
modReg[ 1, "demIntercept" ] <- 1
modReg[ 2, "demPrice" ] <- 1
modReg[ 3, "demIncome" ] <- 1
modReg[ 4, "supIntercept" ] <- 1
modReg[ 5, "supPrice2" ] <- 1
modReg[ 6, "supPrice2" ] <- 1
modReg[ 7, "supTrend" ] <- 1
fitols3 <- systemfit( system, data = Kmenta, restrict.regMat = modReg )
print(coef( fitols3, modified.regMat = TRUE ))
# it seems to me like regMat does the opposite of what it says it does
# in python
# coef1 = np.array([99.8954229, -0.3162988,  0.3346356, 51.9296460,  0.2361566,  0.2361566,  0.2409308])
# i = np.eye(7,6)
# i[-1,-1] = 1
# i[-2,-1] = 0
# i[-2,-2] = 1 
# np.dot(coef,i) # regMat = TRUE?
print(coef( fitols3 ))

### SUR ###
data("GrunfeldGreene")
library(plm)
GGPanel <- plm.data( GrunfeldGreene, c("firm","year") )
formulaGrunfeld <- invest ~ value + capital
greeneSUR <- systemfit( formulaGrunfeld, "SUR", data = GGPanel,
        methodResidCov = "noDfCor" )

#usinvest <- as.matrix(invest[81:100])
#usvalue <- as.matrix(value
col5tbl14_2 <- lm(invest[81:100] ~ value[81:100] + capital[81:100])


