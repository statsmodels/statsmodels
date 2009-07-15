### SETUP ###
d <- read.table("./stata_lbw_glm.csv",sep=",", header=T)
attach(d)
race.f <- factor(race)
contrasts(race.f) <- contr.treatment(3, base = 3)   # make white the intercept

### MODEL ###
m1 <- glm(low ~ age + lwt + race.f + smoke + ptl + ht + ui,
    family=binomial)
results <- summary.glm(m1)
results
results['coefficients']

library(boot)
m1.diag <- glm.diag(m1)
# note that this returns standardized residuals for diagnostics)
