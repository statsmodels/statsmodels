# -*- coding: utf-8 -*-
"""

Created on Mon Dec 10 08:54:02 2012

Author: Josef Perktold
"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa, KappaResults


table0 = np.asarray('''\
1 	0 	0 	0 	0 	14 	1.000
2 	0 	2 	6 	4 	2 	0.253
3 	0 	0 	3 	5 	6 	0.308
4 	0 	3 	9 	2 	0 	0.440
5 	2 	2 	8 	1 	1 	0.330
6 	7 	7 	0 	0 	0 	0.462
7 	3 	2 	6 	3 	0 	0.242
8 	2 	5 	3 	2 	2 	0.176
9 	6 	5 	2 	1 	0 	0.286
10 	0 	2 	2 	3 	7 	0.286'''.split(), float).reshape(10,-1)


Total = np.asarray("20 	28 	39 	21 	32".split('\t'), int)
Pj = np.asarray("0.143 	0.200 	0.279 	0.150 	0.229".split('\t'), float)
kappa_wp = 0.210
table1 = table0[:, 1:-1]


print(fleiss_kappa(table1))
table4 = np.array([[20,5], [10, 15]])
print('res', cohens_kappa(table4), 0.4) #wikipedia

table5 = np.array([[45, 15], [25, 15]])
print('res', cohens_kappa(table5), 0.1304) #wikipedia

table6 = np.array([[25, 35], [5, 35]])
print('res', cohens_kappa(table6), 0.2593)  #wikipedia
print('res', cohens_kappa(table6, weights=np.arange(2)), 0.2593)  #wikipedia
t7 = np.array([[16, 18, 28],
               [10, 27, 13],
               [28, 20, 24]])
print(cohens_kappa(t7, weights=[0, 1, 2]))

table8 = np.array([[25, 35], [5, 35]])
print('res', cohens_kappa(table8))

#SAS example from http://www.john-uebersax.com/stat/saskappa.htm
'''
   Statistic          Value       ASE     95% Confidence Limits
   ------------------------------------------------------------
   Simple Kappa      0.3333    0.0814       0.1738       0.4929
   Weighted Kappa    0.2895    0.0756       0.1414       0.4376
'''
t9 = [[0,  0,  0],
      [5, 16,  3],
      [8, 12, 28]]
res9 = cohens_kappa(t9)
print('res', res9)
print('res', cohens_kappa(t9, weights=[0, 1, 2]))


#check max kappa, constructed by hand, same marginals
table6a = np.array([[30, 30], [0, 40]])
res = cohens_kappa(table6a)
assert res.kappa == res.kappa_max
#print np.divide(*cohens_kappa(table6)[:2])
print(res.kappa / res.kappa_max)


table10 = [[0, 4, 1],
           [0, 8, 0],
           [0, 1, 5]]
res10 = cohens_kappa(table10)
print('res10', res10)


'''SAS result for table10

                  Simple Kappa Coefficient
              --------------------------------
              Kappa                     0.4842
              ASE                       0.1380
              95% Lower Conf Limit      0.2137
              95% Upper Conf Limit      0.7547

                  Test of H0: Kappa = 0

              ASE under H0              0.1484
              Z                         3.2626
              One-sided Pr >  Z         0.0006
              Two-sided Pr > |Z|        0.0011

                   Weighted Kappa Coefficient
              --------------------------------
              Weighted Kappa            0.4701
              ASE                       0.1457
              95% Lower Conf Limit      0.1845
              95% Upper Conf Limit      0.7558

               Test of H0: Weighted Kappa = 0

              ASE under H0              0.1426
              Z                         3.2971
              One-sided Pr >  Z         0.0005
              Two-sided Pr > |Z|        0.0010
'''
