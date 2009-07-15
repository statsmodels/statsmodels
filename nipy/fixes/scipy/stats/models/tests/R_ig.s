### SETUP ###
d <- read.table("./inv_gaussian.csv",sep=",", header=T, nrows=5000)
attach(d)

### MODEL ###
library(nlme)
m1 <- glm(xig ~ x1 + x2, family=inverse.gaussian)
results <- summary.glm(m1)
results
results['coefficients']
