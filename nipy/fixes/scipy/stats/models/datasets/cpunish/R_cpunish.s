### SETUP ###
d <- read.table("./cpunish.csv",sep=",", header=T)
attach(d)
LN_VC100k96 = log(VC100k96*1000)
### MODEL ###
m1 <- glm(EXECUTIONS ~ INCOME + PERPOVERTY + PERBLACK + LN_VC100k96 + SOUTH + DEGREE,
    family=poisson, offset=LN_VC100k96)
results <- summary.glm(m1)
results
results['coefficients']

