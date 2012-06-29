

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
import csv
import nonparametric2 as nparam

import KernelFunctions as kf
import np_tools as tools
reload(nparam)
reload (tools)
reload( kf)
np.random.seed(12345)

NP=importr('np')
r=robjects.r

N=30
o=np.random.binomial(2,0.7,size=(N,1))
o2 = np.random.binomial(3,0.7,size=(N,1))
c1=np.random.normal(size=(N,1))
c2=np.random.normal(2,1,size=(N,1))
c3=np.random.normal(10,2,size=(N,1))

data = csv.reader(open("oecd.csv","r"))
Italy =  csv.reader(open("Italy.csv","r"))
Italy.next()
it_gdp =[]; it_year=[];
for line in Italy:
    it_gdp.append(float(line[1]))
    it_year.append(int(line[0]))

data.next()
growth=[]; year=[]; initgdp = []; popgro=[]; inv=[]; humancap=[]; oecd = []; democracy=[];

for line in data:
    growth.append(float(line[0]))
    oecd.append(int(line[1]))
    year.append(int(line[2]))
    initgdp.append(float(line[3]))
    popgro.append(float(line[4]))
    inv.append(float(line[5]))
    humancap.append(float(line[6]))
    #democracy.append(float(line[7]))
    
#dens_c=nparam.CKDE(tydat=[c1],txdat=[c2], dep_type='c',indep_type='c',bw='cv_ls')
dens_u=nparam.UKDE(tdat=[it_gdp],var_type='c', bw=[ 1.25714112])
#dens_u2=nparam.UKDE(tdat=[it_gdp[0:150], growth[0:150]],var_type='cc', bw=[ 1.37950565,  0.00324568])
#dens = nparam.UKDE(tdat=[growth[0:50], oecd[0:50]], var_type = 'cu', bw ='cv_ls')
#print dens_u2.cdf([20.1, 0.0004])

#dens2=nparam.generic_kde(tdat=[wage,lwage],var_type='cc',bwmethod='normal_reference')

#D={"S1": robjects.FloatVector(c1),"S2":robjects.FloatVector(c2),
#   "S3":robjects.FloatVector(c3), "S4":robjects.FactorVector(o),"S5":robjects.FactorVector(o2)}
#df=robjects.DataFrame(D)
#formula=r('ordered(S4)~ordered(S5)')
#r_bw=NP.npcdensbw(formula, data=df, bwmethod='cv.ls')  #obtain R's estimate of the




print "------------------------"*3
#print 'the estimate by R is: ', r_bw[1],r_bw[0]
#dens_c=nparam.CKDE(tydat=[it_gdp[0:30]],txdat=[it_year[0:30]], dep_type='c',indep_type='o',bw='cv_ls')

#print 'the estimate by SM is: ', dens_c.bw

#print dens.pdf()
