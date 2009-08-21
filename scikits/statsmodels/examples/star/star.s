###################################### SETUP ########################################################
star.data <- as.matrix(read.table("/home/skipper/nipy/skipper-working/nipy/fixes/scipy/stats/models/examples/star/star.bi.dat",header=F))
#star.data <- star.data[-108,]
AVSALK <- star.data[,8]/1000
PERSPENK <- star.data[,9]/1000
star.factors3 <- data.frame( LOWINC=star.data[,2], PERASIAN=star.data[,3], PERBLACK=star.data[,4],
	PERHISP=star.data[,5], PERMINTE=star.data[,6], AVYRSEXP=star.data[,7], AVSAL=AVSALK,
	PERSPEN=PERSPENK, PTRATIO=star.data[,10], PCTAF=star.data[,11], PCTCHRT=star.data[,12],
	PCTYRRND=star.data[,13], READTOT=star.data[,14], PR50RD=star.data[,15],
	MATHTOT=star.data[,16], PR50M=star.data[,17] ) 
attach(star.factors3)
#####################################################################################################

###################################### MATH MODEL ###################################################
star.logit.fit3 <- glm(cbind(PR50M,MATHTOT-PR50M) ~ LOWINC + PERASIAN + PERBLACK + PERHISP +
	PERMINTE * AVYRSEXP * AVSAL + PERSPEN * PTRATIO * PCTAF + PCTCHRT + PCTYRRND,
        family=binomial(link=logit),data=star.factors3)
summary.glm(star.logit.fit3)
anova(star.logit.fit3)

round(cbind(summary.glm(star.logit.fit3)$coef[,1:2],
	summary.glm(star.logit.fit3)$coef[,1]-summary.glm(star.logit.fit3)$coef[,2]*qnorm(.995),
	summary.glm(star.logit.fit3)$coef[,1]+summary.glm(star.logit.fit3)$coef[,2]*qnorm(.995)),8)
