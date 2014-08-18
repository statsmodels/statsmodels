import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.mice import mice
#import csv
##import os
#import numpy as np
#
#cur_dir = os.getcwd()
#fn = os.path.join(cur_dir,"missingdata.csv")

#iternum = 500
#param = []
#se = []
#
#intparam = []
#xparams = []
#intse = []
#xse = []
#for i in range(iternum):        
data = pd.read_csv("missingfull.csv")
data.columns = ['x1','x2','x3']
data = data.drop('x3',1)
impdata = mice.ImputedData(data)
impdata.new_imputer("x2", method="pmm", k_pmm=1)
#impdata.new_imputer("x3", model_class=sm.Poisson, method="pmm", k_pmm=20)
impdata.new_imputer("x1", method="pmm", k_pmm=1, model_class=sm.Logit)
#impdata.new_imputer("x2")
#impdata.new_imputer("x3", model_class=sm.Poisson, method="pmm", k_pmm=20)
#impdata.new_imputer("x1", model_class=sm.Logit)
impcomb = mice.MICE("x1 ~ x2", sm.Logit,impdata)
impcomb.run(2, 1)
p1 = impcomb.combine()
print p1.summary()
#param.append(p1.params)
#se.append(p1.bse)
##    intparam.append(p1.params[0])
##    xparams.append(p1.params[1])
##    intse.append(p1.bse[0])
##    xse.append(p1.bse[1])
#
#t = pd.DataFrame(np.asarray(param))
#t['se1'] = pd.Series(np.asarray(se)[:,0],index=t.index)
#t['se2'] = pd.Series(np.asarray(se)[:,1],index=t.index)
##s = pd.DataFrame(np.asarray(se))
#t.to_csv("pmm_boot.csv", rownames=False)
#s.to_csv("pure_boot_se.csv")
#write = [param,se]
#myfile = open("pure_boot.csv", 'wb')
#wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#wr.writerows(write)