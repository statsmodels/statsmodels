#####################################################################################################
#												    #
#	This archive is part of the free distribution of data and statistical software code for	    #
#	"Generalized Linear Models: A Unified Approach", Jeff Gill, Sage QASS Series.  You are	    #
#	free to use, modify, distribute, publish, etc. provided attribution.  Please forward 	    #
#	bugs, complaints, comments, and useful changes to: jgill@latte.harvard.edu.  		    #
#											    	    #
#####################################################################################################

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
###################################### READING MODEL ################################################
star.logit.fit5 <- glm(cbind(PR50RD,READTOT-PR50RD) ~ LOWINC + PERASIAN + PERBLACK + PERHISP +
	PERMINTE + AVYRSEXP + AVSAL + PERSPEN + PTRATIO + PCTAF + PCTCHRT + PCTYRRND,
        family=binomial(link=logit),data=star.factors3)
star.logit.fit4 <- glm(cbind(PR50RD,READTOT-PR50RD) ~ LOWINC + PERASIAN + PERBLACK + PERHISP +
	PERMINTE * AVYRSEXP * AVSAL + PERSPEN * PTRATIO * PCTAF + PCTCHRT + PCTYRRND,
        family=binomial(link=logit),data=star.factors3)
summary.glm(star.logit.fit4)
anova(star.logit.fit4)

#	summary.glm(star.logit.fit4)$coef[,1]+summary.glm(star.logit.fit4)$coef[,2]*qnorm(.995)),5)

###################################### FIRST DIFFERENCE CALCULATIONS ################################
mean.vector <- apply(star.factors3,2,mean)
diff.vector <- c(1,mean.vector[1:12],mean.vector[5]*mean.vector[6],mean.vector[5]*mean.vector[7],
	mean.vector[6]*mean.vector[7],mean.vector[8]*mean.vector[9],mean.vector[8]*mean.vector[10],
	mean.vector[9]*mean.vector[10],mean.vector[5]*mean.vector[6]*mean.vector[7],
	mean.vector[8]*mean.vector[9]*mean.vector[10])
names(diff.vector) <- names(summary.glm(star.logit.fit4)$coef[,1])
	
# PERMINTE FIRST DIFFERENCE ACROSS IQR
logit(c(diff.vector[1:5],6.329,diff.vector[7:13],6.329*mean.vector[6],6.329*mean.vector[7],
	diff.vector[16:19],6.329*mean.vector[6]*mean.vector[7],diff.vector[21])
	%*%summary.glm(star.logit.fit3)$coef[,1]) - 
logit(c(diff.vector[1:5],19.180,diff.vector[7:13],19.180*mean.vector[6],19.180*mean.vector[7],
	diff.vector[16:19],19.180*mean.vector[6]*mean.vector[7],diff.vector[21])
	%*%summary.glm(star.logit.fit3)$coef[,1])

q1.diff.mat <- matrix(rep(diff.vector,length(diff.vector)),
	nrow=length(diff.vector), ncol=length(diff.vector),
	dimnames=list(names(diff.vector),names(diff.vector)))
diag(q1.diff.mat)[2:13] <- apply(star.factors3,2,summary)[2,1:12]
q1.diff.mat[14,6] <- q1.diff.mat[6,6]*q1.diff.mat[7,6]
q1.diff.mat[15,6] <- q1.diff.mat[6,6]*q1.diff.mat[8,6]
q1.diff.mat[20,6] <- q1.diff.mat[6,6]*q1.diff.mat[7,6]*q1.diff.mat[8,6]

q1.diff.mat[14,7] <- q1.diff.mat[7,7]*q1.diff.mat[6,7]
q1.diff.mat[16,7] <- q1.diff.mat[7,7]*q1.diff.mat[8,7]
q1.diff.mat[20,7] <- q1.diff.mat[6,7]*q1.diff.mat[7,7]*q1.diff.mat[8,7]

q1.diff.mat[15,8] <- q1.diff.mat[8,8]*q1.diff.mat[6,8]
q1.diff.mat[16,8] <- q1.diff.mat[8,8]*q1.diff.mat[7,8]
q1.diff.mat[20,8] <- q1.diff.mat[6,8]*q1.diff.mat[7,8]*q1.diff.mat[8,8]

q1.diff.mat[17,9] <- q1.diff.mat[9,9]*q1.diff.mat[10,9]
q1.diff.mat[18,9] <- q1.diff.mat[9,9]*q1.diff.mat[11,9]
q1.diff.mat[21,9] <- q1.diff.mat[9,9]*q1.diff.mat[10,9]*q1.diff.mat[11,9]

q1.diff.mat[17,10] <- q1.diff.mat[10,10]*q1.diff.mat[9,10]
q1.diff.mat[19,10] <- q1.diff.mat[10,10]*q1.diff.mat[11,10]
q1.diff.mat[21,10] <- q1.diff.mat[9,10]*q1.diff.mat[10,10]*q1.diff.mat[11,10]

q1.diff.mat[18,11] <- q1.diff.mat[11,11]*q1.diff.mat[9,11]
q1.diff.mat[19,11] <- q1.diff.mat[11,11]*q1.diff.mat[10,11]
q1.diff.mat[21,11] <- q1.diff.mat[9,11]*q1.diff.mat[10,11]*q1.diff.mat[11,11]

q2.diff.mat <- matrix(rep(diff.vector,length(diff.vector)),
        nrow=length(diff.vector), ncol=length(diff.vector),
        dimnames=list(names(diff.vector),names(diff.vector)))
diag(q2.diff.mat)[2:13] <- apply(star.factors3,2,summary)[5,1:12]
q2.diff.mat[14,6] <- q2.diff.mat[6,6]*q2.diff.mat[7,6]
q2.diff.mat[15,6] <- q2.diff.mat[6,6]*q2.diff.mat[8,6]
q2.diff.mat[20,6] <- q2.diff.mat[6,6]*q2.diff.mat[7,6]*q2.diff.mat[8,6]

q2.diff.mat[14,7] <- q2.diff.mat[7,7]*q2.diff.mat[6,7]
q2.diff.mat[16,7] <- q2.diff.mat[7,7]*q2.diff.mat[8,7]
q2.diff.mat[20,7] <- q2.diff.mat[6,7]*q2.diff.mat[7,7]*q2.diff.mat[8,7]

q2.diff.mat[15,8] <- q2.diff.mat[8,8]*q2.diff.mat[6,8]
q2.diff.mat[16,8] <- q2.diff.mat[8,8]*q2.diff.mat[7,8]
q2.diff.mat[20,8] <- q2.diff.mat[6,8]*q2.diff.mat[7,8]*q2.diff.mat[8,8]

q2.diff.mat[17,9] <- q2.diff.mat[9,9]*q2.diff.mat[10,9]
q2.diff.mat[18,9] <- q2.diff.mat[9,9]*q2.diff.mat[11,9]
q2.diff.mat[21,9] <- q2.diff.mat[9,9]*q2.diff.mat[10,9]*q2.diff.mat[11,9]

q2.diff.mat[17,10] <- q2.diff.mat[10,10]*q2.diff.mat[9,10]
q2.diff.mat[19,10] <- q2.diff.mat[10,10]*q2.diff.mat[11,10]
q2.diff.mat[21,10] <- q2.diff.mat[9,10]*q2.diff.mat[10,10]*q2.diff.mat[11,10]

q2.diff.mat[18,11] <- q2.diff.mat[11,11]*q2.diff.mat[9,11]
q2.diff.mat[19,11] <- q2.diff.mat[11,11]*q2.diff.mat[10,11]
q2.diff.mat[21,11] <- q2.diff.mat[9,11]*q2.diff.mat[10,11]*q2.diff.mat[11,11]

for (i in 2:12)  
	print( paste( names(diff.vector)[i],
	round(logit(q2.diff.mat[,i]%*%summary.glm(star.logit.fit3)$coef[,1]) - 
	logit(q1.diff.mat[,i]%*%summary.glm(star.logit.fit3)$coef[,1]),4) ) )
	


###################################### COPLOTS FOR INTERACTIONS #####################################
postscript("Book.GLM/Example.STAR/star.coplot1.ps")
coplot.jg((PR50M/MATHTOT)~PERMINTE|AVYRSEXP*AVSAL)
dev.off()

postscript("Book.GLM/Example.STAR/star.coplot2.ps")
coplot.jg((PR50M/MATHTOT)~PERSPEN|PTRATIO*PCTAF)
dev.off()

source("Book.GLM/Example.STAR/coplot.jg")
postscript("Book.GLM/Example.STAR/star.coplot3.ps")
coplot.jg((PR50RD/READTOT)~PERMINTE|AVYRSEXP*AVSAL)
dev.off()


mean(PR50RD/READTOT - PR50M/MATHTOT)


###################################### RESIDUALS ANALYSIS ###########################################

star.pears <- resid(star.logit.fit,type="pearson")
star.pears%*%star.pears
star.mat <- cbind(rep(1,nrow(star.factors)),as.matrix(star.factors[1:12]))
star.mu <- predict.glm(star.logit.fit,type="response")
star.y <- PR50M/MATHTOT
star.n <- length(star.y)

sum(resid(star.logit.fit,type="response")) 
sum(resid(star.logit.fit,type="pearson")) 

star.phi <- cs.string

# GET ADJUSTED DEVIANCES
2*sum(  PR50M*log(star.y/star.mu) + (MATHTOT-PR50M)*log((MATHTOT-PR50M)/(MATHTOT-star.mu*MATHTOT)) )
[1] 4078.765

mean(PR50M)	[1] 357.8152
PR50M.adj <- PR50M
for (i in 1:length(PR50M.adj))  {
	if (PR50M.adj[i] > mean(PR50M)) PR50M.adj[i] <- PR50M.adj[i] - 0.5
	if (PR50M.adj[i] < mean(PR50M)) PR50M.adj[i] <- PR50M.adj[i] + 0.5
}

2*sum(  PR50M.adj*log(PR50M.adj/(star.mu*MATHTOT)) 
        + (MATHTOT-PR50M.adj)*log((MATHTOT-PR50M.adj)/(MATHTOT-star.mu*MATHTOT)) )
[1] 4054.928

range(abs( eigen(-glm.vc(star.logit.fit))$values^(-1) ))

###################################### FIGURE 1: MODEL FIT DIAGNOSTICS ##############################
postscript("/export/home/jgill/Book.GLM/glm.fig1.ps")#,width=3,height=3)
par(mfrow=c(1,3),mar=c(8,8,6,2))
plot(star.mu,star.y,xlab="",ylab="")
mtext(side=1,outer=F,"Fitted Values",line=6,cex=1.5)
mtext(side=2,outer=F,"Observed Values",line=4,cex=1.3)
mtext(side=3,outer=F,"Model Fit Plot",line=3,cex=1.4)
abline(lm(star.y~star.mu)$coefficients)
plot(fitted(star.logit.fit),resid(star.logit.fit,type="pearson"),xlab="",ylab="")
mtext(side=1,outer=F,"Fitted Values",line=6,cex=1.5)
mtext(side=2,outer=F,"Pearson Residuals",line=4,cex=1.3)
mtext(side=3,outer=F,"Residual Dependence Plot",line=3,cex=1.4)
abline(0,0)
qqnorm(resid(star.logit.fit,type="deviance"),main="",xlab="",ylab="")
mtext(side=1,outer=F,"Quantiles of N(0,1)",line=6,cex=1.5)
mtext(side=2,outer=F,"Deviance Residual Quantiles",line=4,cex=1.3)
mtext(side=3,outer=F,"Normal-Quantile Plot",line=3,cex=1.4)
abline(-0.3,3.5)
dev.off()

#####################################################################################################


