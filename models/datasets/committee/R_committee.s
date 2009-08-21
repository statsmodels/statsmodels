### SETUP ###
d <- read.table("./committee.csv",sep=",", header=T)
attach(d)

LNSTAFF <- log(STAFF)
SUBS.LNSTAFF <- SUBS*LNSTAFF
library(MASS)
#m1 <- glm.nb(BILLS104 ~ SIZE + SUBS + LNSTAFF + PRESTIGE + BILLS103 + SUBS.LNSTAFF)
m1 <- glm(BILLS104 ~ SIZE + SUBS + LNSTAFF + PRESTIGE + BILLS103 + SUBS.LNSTAFF, family=negative.binomial(1))  # Disp should be 1 by default

results <- summary.glm(m1)
