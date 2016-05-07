### SETUP ###
d <- read.table("./cpunish.csv",sep=",", header=T)
attach(d)
LN_VC100k96 = log(VC100k96)
### MODEL ###
m1 <- glm(EXECUTIONS ~ INCOME + PERPOVERTY + PERBLACK + LN_VC100k96 + SOUTH + DEGREE,
    family=poisson)
results <- summary.glm(m1)
results
results['coefficients']

# Model with exposure
m2 <- glm(EXECUTIONS ~ INCOME + PERPOVERTY + PERBLACK + LN_VC100k96 + SOUTH + DEGREE,
    family=poisson, offset=rep(log(100), length(EXECUTIONS)))
results2 <- summary.glm(m2)
