### SETUP
#star.data <- as.matrix(read.csv("./star98.csv",header=T))
#star.factors3 <- data.frame( LOWINC=star.data[,3], PERASIAN=star.data[,4], PERBLACK=star.data[,5],
#	PERHISP=star.data[,6], PERMINTE=star.data[,7], AVYRSEXP=star.data[,8], AVSAL=star.data[,9],
#	PERSPEN=star.data[,10], PTRATIO=star.data[,11], PCTAF=star.data[,12], PCTCHRT=star.data[,13],
#	PCTYRRND=star.data[,14], PERMINTE.AVYRSEXP=star.data[,15], PERMINTE.AVSAL=star.data[,16],
#    AVYRSEXP.AVSAL=star.data[,17], PERSPEN.PTRATIO=star.data[,18], PERSPEN.PCTAF=star.data[,19],
#    PTRATIO.PCTAF=star.data[,20], PERMINTE.AVYRSEXP.AVSAL=star.data[,21],
#    PERSPEN.PTRATIO.PCTAF=star.data[,22], MATHTOT=star.data[,1], PR50M=star.data[,2] )
d <- read.table("./star98.csv", sep=",", header=T)
attach(d)
#attach(star.factors3)


### MATH MODEL
m1 <-  glm(cbind(PR50M,MATHTOT-PR50M) ~ LOWINC + PERASIAN + PERBLACK + PERHISP +
    PERMINTE + AVYRSEXP + AVSALK + PERSPENK + PTRATIO + PCTAF + PCTCHRT + PCTYRRND +
    PERMINTE_AVYRSEXP + PERMINTE_AVSAL + AVYRSEXP_AVSAL + PERSPEN_PTRATIO + PERSPEN_PCTAF + 
    PTRATIO_PCTAF + PERMINTE_AVYRSEXP_AVSAL + PERSPEN_PTRATIO_PCTAF,
    family=binomial)
#as.numeric(m1$coef)
#as.numeric(sqrt(diag(vcov(m1))))
results <- summary.glm(m1)
    
#star.logit.fit3 <- glm(cbind(PR50M,MATHTOT-PR50M) ~ LOWINC + PERASIAN + PERBLACK + PERHISP +
#    PERMINTE + AVYRSEXP + AVSAL + PERSPEN + PTRATIO + PCTAF + PCTCHRT + PCTYRRND +
#    PERMINTE.AVYRSEXP + PERMINTE.AVSAL + AVYRSEXP.AVSAL + PERSPEN.PTRATIO + PERSPEN.PCTAF + 
#    PTRATIO.PCTAF + PERMINTE.AVYRSEXP.AVSAL + PERSPEN.PTRATIO.PCTAF, 
#    family = binomial(), data=star.factors3)
#results <- summary.glm(star.logit.fit3)
# WITH R STYLE INTERACTIONS
#star.logit.fit4 <- glm(cbind(PR50M,MATHTOT-PR50M) ~ LOWINC + PERASIAN + PERBLACK + PERHISP +
#    PERMINTE + AVYRSEXP + AVSAL + PERSPEN + PTRATIO + PCTAF + PCTCHRT + PCTYRRND +
#    PERMINTE*AVYRSEXP*AVSAL + PERSPEN*PTRATIO*PCTAF, 
#    family = binomial(), data=star.factors3)


