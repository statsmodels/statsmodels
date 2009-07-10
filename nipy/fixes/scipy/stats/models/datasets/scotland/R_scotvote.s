### SETUP ###
d <- read.table("./scotvote.csv",sep=",", header=T)
attach(d)

### MODEL ###
m1 <- glm(YES ~ COUTAX * UNEMPF + MOR + ACT + GDP + AGE,
    family=Gamma)
results <- summary.glm(m1)
results
results['coefficients']
