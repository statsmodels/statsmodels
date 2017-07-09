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
#Exploratory factor analysis is only useful if the matrix of population 
#correlation is statistically different from the identity matrix. 
#If these are equal, the variables are few interrelated, i.e., the specific 
#factors explain the greater proportion of the variance and the common factors
#are unimportant. Therefore, it should be defined when the correlations 
#between the original variables are sufficiently high. 
#Thus, the factor analysis is useful in estimation of common factors. 
#With this in mind, the Bartlett Sphericity test can be used. The hypotheses are:

# H0: the matrix of population correlations is equal to the identity matrix
# H1: the matrix of population correlations is different from the identity matrix.

def bartlett_sphericity(dataset, corr_method="pearson"):
    
    r"""
    
    Parameters
    ----------
    dataset : dataframe, mandatory (numerical or ordinal variables)
        
    corr_method : {'pearson', 'spearman'}, optional
        
    Returns
    -------
    out : namedtuple
        The function outputs the test value (chi2), the degrees of freedom (ddl)
        and the p-value.
        It also delivers the n_p_ratio if the number of instances (n) divided 
        by the numbers of variables (p) is more than 5. A warning might be issued.
        
        Ex:
        chi2:  410.27280642443156
        ddl:  45.0
        p-value:  8.73359410503e-61
        n_p_ratio:    20.00
        
        Out: Bartlett_Sphericity_Test_Results(chi2=410.27280642443156, ddl=45.0, pvalue=8.7335941050291506e-61)
    
    References
    ----------
    
    [1] Bartlett,  M.  S.,  (1951),  The  Effect  of  Standardization  on  a  chi  square  Approximation  in  Factor
    Analysis, Biometrika, 38, 337-344.
    [2] R. Sarmento and V. Costa, (2016)
    "Comparative Approaches to Using R and Python for Statistical Data Analysis"
    in press, Cybertech Publishing.
    
    Examples
    --------
    illustration how to use the function.
    
    >>> bartlett_sphericity(survey_data, corr_method="spearman")
    chi2:  410.27280642443156
    ddl:  45.0
    p-value:  8.73359410503e-61
    n_p_ratio:    20.00
    C:\Users\Rui Sarmento\Anaconda3\lib\site-packages\spyderlib\widgets\externalshell\start_ipython_kernel.py:75: 
    UserWarning: NOTE: we advise  to  use  this  test  only  if  the number of instances (n) divided by the number of variables (p) is lower than 5. Please try the KMO test, for example.
    backend_o = CONF.get('ipython_console', 'pylab/backend', 0)
    Out[12]: Bartlett_Sphericity_Test_Results(chi2=410.27280642443156, ddl=45.0, pvalue=8.7335941050291506e-61)
    """
    
    import numpy as np
    import math as math
    import scipy.stats as stats
    import warnings as warnings
    import collections

    #Dimensions of the Dataset
    n = dataset.shape[0]
    p = dataset.shape[1]
    n_p_ratio = n / p
    
    #Several Calculations
    chi2 = - (n - 1 - (2 * p + 5) / 6) * math.log(np.linalg.det(dataset.corr(method=corr_method)))
    #Freedom Degree
    ddl = p * (p - 1) / 2
    #p-value
    pvalue = stats.chi2.pdf(chi2 , ddl)
    
    Result = collections.namedtuple("Bartlett_Sphericity_Test_Results", ["chi2", "ddl", "pvalue"], verbose=False, rename=False)   
    
    #Output of the results - named tuple
    result = Result(chi2=chi2,ddl=ddl,pvalue=pvalue) 

    
    #Output of the function
    if n_p_ratio > 5 :
        print("n_p_ratio: {0:8.2f}".format(n_p_ratio))
        warnings.warn("NOTE: we advise  to  use  this  test  only  if  the number of instances (n) divided by the number of variables (p) is lower than 5. Please try the KMO test, for example.")
        
    
    return result