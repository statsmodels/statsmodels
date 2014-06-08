# -*- coding: utf-8 -*-
"""
Created on Thu May 08 04:39:24 2014

@author: Frank
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 07 15:10:00 2014

@author: Frank
"""

import sys
import pandas as pd
import numpy as np
sys.path.insert(0,"C:/Users/Frank/Dropbox/statsmodels/")
#sys.path.insert(0,"C:/Users/Frank/statsmodels/statsmodels/")
from statsmodels.sandbox.mice import mice
import matplotlib.pyplot as plt
import statsmodels.api as sm
import csv
import json
#print statsmodels.__file__
#data = pd.DataFrame.from_csv('missingdata.csv')
paramspmm = []
sdepmm = []
for i in range(500):
        
    data = pd.read_csv("C:/Users/Frank/Dropbox/statsmodels/statsmodels/sandbox/mice/tests/results/missingdata.csv")
    #data = pd.read_csv('C:/Users/Frank/statsmodels/statsmodels/sandbox/mice/tests/results/missingdata.csv')
    #data = np.genfromtxt('missingdata.csv',delimiter = ',')
    data.columns = ['x1','x2','x3']
    impdata = mice.ImputedData(data)
    
    #print(impdata.data.fillna(impdata.data.mean()))
    #m1 = mice.Imputer("x2 ~ x1 + x3",sm.OLS, impdata, scale = "perturb_chi2")
    #
    #m2 = mice.Imputer("x3 ~ x1 + x2",sm.OLS, impdata, scale = "perturb_chi2")
    #
    #m3 = mice.Imputer("x1 ~ x2 + x3",sm.Logit, impdata)
    
    m1 = impdata.new_imputer("x2")
    m2 = impdata.new_imputer("x3")
    m3 = impdata.new_imputer("x1", model_class=sm.Logit)
    
    
    impdata = mice.ImputedData(data)
    
    #impchain = mice.ImputerChain([m1,m2,m3])
    
    #test = sm.Logit(data['x1'],data['x2'],'drop')
    
    impcomb = mice.MICE("x1 ~ x2 + x3", sm.Logit,[m1,m2,m3])
    
    implist = impcomb.run()
    
    p1 = impcomb.combine(implist)
    paramspmm.append(p1.params)
    sdepmm.append(p1.bse)
    #impdata = mice.ImputedData(data)
    #
    #impfull = mice.AnalysisChain([m1,m2,m3], "x1 ~ x2 + x3", sm.Logit)
    ##
    #p2 = impfull.run_chain(20,10)
    #
    #p1.summary()
    #p2.summary()
    
    #make some graphs of imputed pmm versus abayes
    
    #print p1
    #print p2
    #print s1
    #print s2
    #impchain.generate_data(3,5,'ftest')
    #
    #
    #
    #
    #f0 = pd.read_csv('ftest_0.csv')
    #f1 = pd.read_csv('ftest_1.csv')
    #f2 = pd.read_csv('ftest_2.csv')
    #
    #s1 = set(impdata.values['x2'][0])
    #s2 = set(impdata.values['x3'][0])
    #mval = s1.union(s2)
    #mvalb = s1.intersection(s2)
    #
    #f0c = []
    #f0both = []
    #f0a = []
    #f0b = []
    #for i in range(len(f0.x2)):
    #    f0c.append(i not in mval)
    #    f0both.append(i in mvalb)
    #    f0a.append(i in s1)
    #    f0b.append(i in s2)
    #
    #colors = np.where(f0c,'r','k')
    #
    #plt.scatter(f0.x2, f0.x3, s=120, c=colors)
    #
    #f0['subset'] = np.select([f0a, f0b, f0c],
    #                         ['x2 missing','x3 missing', 'not missing'])
    #for color, label in zip('bgr',  ['x2 missing','x3 missing', 'not missing']):
    #    subset = f0[f0.subset == label]
    #    plt.scatter(subset.x2, subset.x3, s=120, c=color, label=str(label))
#    ##plt.legend()
#myfile=open("pparams_bayes.csv",'wb')    
#wr = csv.writer(myfile,quoting=csv.QUOTE_ALL)
#wr.writerow(params2)
#
#
#myfile=open("ppstd_bayes.csv",'wb')    
#wr = csv.writer(myfile,quoting=csv.QUOTE_ALL)
#wr.writerow(sde2)
#
#with open('C:/Users/Frank/Dropbox/statsmodels/statsmodels/sandbox/mice/tests/pparams_bayes.csv','Ur') as f:
#    data = list(list(rec) for rec in csv.reader(f,lineterminator='\n',delimiter='\t'))