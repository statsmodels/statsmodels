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

#KMO Test

def kmo_test(dataset_corr):
    
    r"""
    
    Parameters
    ----------
    dataset_corr : correlation matrix
        
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
    
    >>>kmo_test(survey_data.corr(method="spearman"))
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
"""
    
    import numpy as np
    import math as math
    import collections

    #KMO Test
    #inverse of the correlation matrix
    invR = np.linalg.inv(dataset_corr)
    nrow_invR = dataset_corr.shape[0]
    ncol_invR = dataset_corr.shape[1]
    
    #partial correlation matrix
    A = np.matrix(np.ones((nrow_invR,ncol_invR)))
    for i in range(0,nrow_invR,1):
        for j in range(i,ncol_invR,1):
            #above the diagonal
            A.itemset((i,j),round(-(invR[i,j]) / (math.sqrt(invR[i,i]*invR[j,j])),9)) 
            #below the diagonal
            A.itemset((j,i),A[i,j])
    
    
    #KMO value
    kmo_num = np.sum(np.square(dataset_corr))-(np.sum(np.diagonal(np.square(dataset_corr))))
    kmo_denom = kmo_num + np.sum(np.square(A))-(np.sum(np.diagonal(np.square(A))))
    kmo_value = kmo_num/kmo_denom
    #print(kmo_value)
    
    #transform to an array of arrays ("matrix" with Python)
    dataset_corr = np.matrix(dataset_corr)
    
    #KMO per variable (diagonal of the spss anti-image matrix)
    for j in range(0, dataset_corr.shape[1],1):
        kmo_j_num = np.sum(np.square(dataset_corr[:,[j]]))-dataset_corr[j,j]**2
        kmo_j_denom = kmo_j_num + (np.sum(np.square(A[:,[j]]))-A[j,j]**2)
        kmo_j = kmo_j_num / kmo_j_denom
        #print(kmo_j)

    
    Result = collections.namedtuple("KMO_Test_Results", ["value", "per_variable"], verbose=False, rename=False)   
    
    #Output of the results - named tuple
    result = Result(value=kmo_j,per_variable=kmo_value)
    
    return result