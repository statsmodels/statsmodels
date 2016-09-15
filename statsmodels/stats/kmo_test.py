# -*- coding: utf-8 -*-
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

import numpy as np
import math as math
import collections

#KMO Test
def kmo_test(dataset_corr):
    
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
        KMO_Test_Results(value=0.85649724257367099, per_variable=Q1     1.275049
        Q2     1.250335
        Q3     1.252462
        Q4     1.255828
        Q5     1.278402
        Q6     1.263415
        Q7     1.251248
        Q8     1.260742
        Q9     1.267690
        Q10    1.256992
        dtype: float64)
    
    References
    ----------    
    [1] Kaiser, H. F. (1970). A second generation little jiffy. Psychometrika, 35(4), 401-415.
    [2] Kaiser, H. F. (1974). An index of factorial simplicity. Psychometrika, 39(1), 31-36.
    [3] R. Sarmento and V. Costa, (2016)
    "Comparative Approaches to Using R and Python for Statistical Data Analysis"
    in press, Cybertech Publishing.
    
    Examples
    --------
    illustration how to use the function.
    
    >>> kmo_test(survey_data.corr(method="spearman"))
         
        KMO_Test_Results(value=0.85649724257367099, per_variable=Q1     1.275049
        Q2     1.250335
        Q3     1.252462
        Q4     1.255828
        Q5     1.278402
        Q6     1.263415
        Q7     1.251248
        Q8     1.260742
        Q9     1.267690
        Q10    1.256992
        dtype: float64) 
"""
    
    

    #KMO Test
    #inverse of the correlation matrix
    invR = np.linalg.inv(dataset_corr)
    nrow_invR, ncol_invR = dataset_corr.shape
    
    #partial correlation matrix
    A = np.ones((nrow_invR,ncol_invR))
    for i in range(0,nrow_invR,1):
        for j in range(i,ncol_invR,1):
            #above the diagonal
            A[i,j] = round(-(invR[i,j]) / (math.sqrt(invR[i,i]*invR[j,j])),9) 
            #below the diagonal
            A[j,i] = A[i,j]
    
    
    #KMO value
    kmo_num = np.sum(np.square(dataset_corr))-(np.sum(np.diagonal(np.square(dataset_corr))))
    kmo_denom = kmo_num + np.sum(np.square(A))-(np.sum(np.diagonal(np.square(A))))
    kmo_value = kmo_num/kmo_denom
    
    #transform to an array of arrays ("matrix" with Python)
    dataset_corr = np.asarray(dataset_corr)
    
    #KMO per variable (diagonal of the spss anti-image matrix)
    for j in range(0, dataset_corr.shape[1]):
        kmo_j_num = np.sum((dataset_corr[:,[j]])**2)-dataset_corr[j,j]**2
        kmo_j_denom = kmo_j_num + (np.sum((A[:,[j]])**2)-A[j,j]**2)
        kmo_j = kmo_j_num / kmo_j_denom

    
    Result = collections.namedtuple("KMO_Test_Results", ["value", "per_variable"])   
    
    #Output of the results - named tuple    
    return Result(value=kmo_j,per_variable=kmo_value)
