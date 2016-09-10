# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 20:21:04 2016

Bartlett Sphericity Test Function:

- This test evaluates sampling adequacy for exploratory Factor Analysis

Bartlett_Sphericity function has two inputs:

- The Dataset (numerical or ordinal variables only)
- The correlation method (spearman or pearson)

It Outputs the test result, degrees of freedom and p-value


@authors: Rui Sarmento
          Vera Costa
"""

#Bartlett Sphericity Test

def Bartlett_Sphericity(dataset, corr_method="pearson"):
    
    import numpy as np
    import math as math
    import scipy.stats as stats

    #Dimensions of the Dataset
    n = dataset.shape[0]
    p = dataset.shape[1]
    n_p_ratio = n/p
    
    #Generate Identity Matrix (pxp)
    indentity = np.identity(p)
    
    #Several Calculations
    chi2 = -(n-1-(2*p+5)/6)*math.log(np.linalg.det(dataset.corr(method=corr_method)))
    #Freedom Degree
    ddl = p*(p-1)/2
    #p-value
    pvalue = stats.chi2.pdf(chi2 , ddl)
    
    #Output of the function
    print("chi2: ", chi2)
    print("ddl: ", ddl)
    print("p-value: ", pvalue)
    if n_p_ratio > 5 :
        print("n_p_ratio: {0:8.2f}".format(n_p_ratio))
        print("Warning: we advise  to  use  this  test  only  if  the number of instances divided by the number of variables is lower than 5. Please try the KMO test, for example.")
        
    
    return {}