### SETUP ###
d <- read.table("./scotvote.csv",sep=",", header=T)
attach(d)

### MODEL ###
m1 <- glm(YES ~ COUTAX * UNEMPF + MOR + ACT + GDP + AGE,
    family=Gamma)
results <- summary.glm(m1)
results
results['coefficients']
logLik(m1)
scale <- results$disp
Y <- YES
mu <- m1$fitted
llf <- -1/scale * sum(Y/mu+log(mu)+(scale-1)*log(Y)+log(scale)+scale*lgamma(1/scale))
print(llf)
print("This is the llf calculated with the formula")
