

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
import csv
import nonparametric2 as nparam
import KernelFunctions as kf




NP=importr('np')
r=robjects.r

N=300
u=np.random.binomial(2,0.7,size=(N,1))
c1=np.random.normal(size=(N,1))
c2=np.random.normal(2,1,size=(N,1))
c3=np.random.normal(3,2,size=(N,1))


dens_c=nparam.CKDE(tydat=[c1,c3],txdat=[c2], dep_type='cc',indep_type='c',bw='normal_reference')
dens_u=nparam.UKDE(tdat=[c1,c2],var_type='cc',bw='cv_ml')
#dens2=nparam.generic_kde(tdat=[wage,lwage],var_type='cc',bwmethod='normal_reference')

D={"S1": robjects.FloatVector(c1),"S2":robjects.FloatVector(c2),"S3":robjects.FloatVector(c3)}
df=robjects.DataFrame(D)
formula=r('~S1+S2')
#r_bw=NP.npudensbw(formula, data=df, bwmethod='cv.ml')  #obtain R's estimate of the
r_bw=NP.npudensbw(formula, data=df, bwmethod='cv.ml')



print "------------------------"*4
print 'the estimate by R is: ', r_bw[1],r_bw[0], '||||||', 'the estimate by SM is: ', dens_u.bw

#print dens.pdf()
