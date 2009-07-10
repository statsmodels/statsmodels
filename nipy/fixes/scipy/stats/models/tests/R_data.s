### SETUP ###
d <- read.table("./datafile.csv",sep=",", header=F)
attach(d)

### MODEL ###
m1 <- glm(V1 ~ V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15,
    family=gaussian)
results <- summary.glm(m1)
results
results['coefficients']
