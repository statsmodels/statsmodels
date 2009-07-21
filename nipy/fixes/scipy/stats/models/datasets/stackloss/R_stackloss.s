### SETUP ###
d <- read.table("./stackloss.csv",sep=",", header=T)
attach(d)
library(MASS)


m1 <- rlm(STACKLOSS ~ AIRFLOW + WATERTEMP + ACIDCONC) # psi.huber default

m2 <- rlm(STACKLOSS ~ AIRFLOW + WATERTEMP + ACIDCONC, psi = psi.hampel, init = "lts")

m3 <- rlm(STACKLOSS ~ AIRFLOW + WATERTEMP + ACIDCONC, psi = psi.bisquare)

results1 <- summary(m1)

results2 <- summary(m2)

results3 <- summary(m3)

m4 <- rlm(STACKLOSS ~ AIRFLOW + WATERTEMP + ACIDCONC, scale.est="Huber") # psi.huber default

m5 <- rlm(STACKLOSS ~ AIRFLOW + WATERTEMP + ACIDCONC, scale.est="Huber", psi = psi.hampel, init = "lts")

m6 <- rlm(STACKLOSS ~ AIRFLOW + WATERTEMP + ACIDCONC, scale.est="Huber", psi = psi.bisquare)

results4 <- summary(m4)

results5 <- summary(m5)

results6 <- summary(m6)

