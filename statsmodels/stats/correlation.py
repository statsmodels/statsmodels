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
    [2] R. Sarmento and V. Costa, (2017)
    "Comparative Approaches to Using R and Python for Statistical Data Analysis", IGI-Global.
    
    Examples
    --------
    illustration how to use the function.
    
    >>> bartlett_sphericity(survey_data, corr_method="spearman")
    chi2:  410.27280642443145
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
    

"""
Created on Sat Sep 10 20:21:04 2016

KMO Test Function:

- This test evaluates sampling adequacy for exploratory Factor Analysis

KMO Test function has one input:

- The Dataset Correlation Matrix

It Outputs the test result, and the results per variable


@authors: Rui Sarmento
          Vera Costa
"""

#KMO Test
#KMO is a measure of the adequacy of sampling “Kaiser-Meyer-Olkin" and checks 
#if it is possible to factorize the main variables efficiently.
#The correlation matrix is always the starting point. The variables are more or
#less correlated, but the others can influence the correlation between the two 
#variables. Hence, with KMO, the partial correlation is used to measure the 
#relation between two variables by removing the effect of the remaining variables.

def kmo(dataset_corr):
    
    import numpy as np
    import math as math
    import collections
    
    r"""
    
    Parameters
    ----------
    dataset_corr : ndarray
        Array containing dataset correlation
        
    Returns
    -------
    out : namedtuple
        The function outputs the test value (value), the test value per variable (per_variable)
       
        Ex:
        Out[30]: 
        KMO_Test_Results(value=0.798844102413, 
        per_variable=
        Q1     0.812160468405
        Q2     0.774161264483
        Q3     0.786819432663
        Q4     0.766251123086
        Q5     0.800579196084
        Q6     0.842927745203 
        Q7     0.792010173432 
        Q8     0.862037322891
        Q9     0.714795031915 
        Q10    0.856497242574
        dtype: float64)
    
    References
    ----------    
    [1] Kaiser, H. F. (1970). A second generation little jiffy. Psychometrika, 35(4), 401-415.
    [2] Kaiser, H. F. (1974). An index of factorial simplicity. Psychometrika, 39(1), 31-36.
    [3] R. Sarmento and V. Costa, (2017)
    "Comparative Approaches to Using R and Python for Statistical Data Analysis", IGI-Global
    
    Examples
    --------
    illustration how to use the function.
    
    >>> kmo_test(survey_data.corr(method="spearman"))
         
        KMO_Test_Results(value=0.798844102413, 
        per_variable=
        Q1     0.812160468405
        Q2     0.774161264483
        Q3     0.786819432663
        Q4     0.766251123086
        Q5     0.800579196084
        Q6     0.842927745203 
        Q7     0.792010173432 
        Q8     0.862037322891
        Q9     0.714795031915 
        Q10    0.856497242574
        dtype: float64) 
"""
    
    

    #KMO Test
    #inverse of the correlation matrix
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    
    #partial correlation matrix
    A = np.ones((nrow_inv_corr,ncol_inv_corr))
    for i in range(0,nrow_inv_corr,1):
        for j in range(i,ncol_inv_corr,1):
            #above the diagonal
            A[i,j] = - (corr_inv[i,j]) / (math.sqrt(corr_inv[i,i] * corr_inv[j,j]))
            #below the diagonal
            A[j,i] = A[i,j]
    
    #transform to an array of arrays ("matrix" with Python)
    dataset_corr = np.asarray(dataset_corr)
        
    #KMO value
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(dataset_corr)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    
    
    kmo_j = [None]*dataset_corr.shape[1]
    #KMO per variable (diagonal of the spss anti-image matrix)
    for j in range(0, dataset_corr.shape[1]):
        kmo_j_num = np.sum(dataset_corr[:,[j]] ** 2) - dataset_corr[j,j] ** 2
        kmo_j_denom = kmo_j_num + np.sum(A[:,[j]] ** 2) - A[j,j] ** 2
        kmo_j[j] = kmo_j_num / kmo_j_denom

    
    Result = collections.namedtuple("KMO_Test_Results", ["value", "per_variable"])   
    
    #Output of the results - named tuple    
    return Result(value=kmo_value,per_variable=kmo_j)
