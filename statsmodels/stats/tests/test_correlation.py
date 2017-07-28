
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:32:42 2016

@author: Rui Sarmento
"""

from statsmodels.stats.descriptivestats import sign_test
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd

def test_bartlett_sphericity():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = pd.DataFrame.from_csv(os.path.join(cur_dir, 'results', "tests.csv"),header=None,sep=',',index_col=None)
      
     
    chi2,ddl,pvalue = bartlett_sphericity(dataset, corr_method="spearman")
    # from R cortest.bartlett(correlation, n=nrow(data))
    # from psych package
    
    # from R
    assert_almost_equal(pvalue, 1.968257e-60, 5)
    # from R
    assert_almost_equal(chi2, 410.272, 1)
    # from R
    assert_equal(ddl, 45)
    

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:32:42 2016

@author: Rui Sarmento
"""

from statsmodels.stats.descriptivestats import sign_test
from numpy.testing import assert_almost_equal, assert_equal

def test_kmo():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = pd.DataFrame.from_csv(os.path.join(cur_dir, 'results', "tests.csv"),header=None,sep=',',index_col=None)
      
    dataset_corr = dataset.corr(method="spearman")
     
    value,per_variable = kmo(dataset_corr)
    # from R cortest.bartlett(correlation, n=nrow(data))
    # from psych package
    
    # from R
    assert_almost_equal(value, 0.8, 1)
    # from R
    assert_almost_equal(per_variable, [0.81,0.77,0.79,0.77,0.80,0.84,0.79,0.86,0.71,0.86],2)