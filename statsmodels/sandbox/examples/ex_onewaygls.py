"""Example: Test for equality of coefficients across groups/regressions


Created on Sat Mar 27 22:36:51 2010
Author: josef-pktd
"""

import numpy as np
from scipy import stats

#from numpy.testing import assert_almost_equal
import statsmodels.api as sm
from statsmodels.sandbox.regression.onewaygls import OneWayLS

#choose example
#--------------
example = ['null', 'diff'][1]   #null: identical coefficients across groups
example_size = [10, 100][0]
example_size = [(10,2), (100,2)][0]
example_groups = ['2', '2-2'][1]
#'2-2': 4 groups,
#       groups 0 and 1 and groups 2 and 3 have identical parameters in DGP

#generate example
#----------------
np.random.seed(87654589)
nobs, nvars = example_size
x1 = np.random.normal(size=(nobs, nvars))
y1 = 10 + np.dot(x1,[15.]*nvars) + 2*np.random.normal(size=nobs)

x1 = sm.add_constant(x1, prepend=False)
#assert_almost_equal(x1, np.vander(x1[:,0],2), 16)
#res1 = sm.OLS(y1, x1).fit()
#print res1.params
#print np.polyfit(x1[:,0], y1, 1)
#assert_almost_equal(res1.params, np.polyfit(x1[:,0], y1, 1), 14)
#print res1.summary(xname=['x1','const1'])

#regression 2
x2 = np.random.normal(size=(nobs,nvars))
if example == 'null':
    y2 = 10 + np.dot(x2,[15.]*nvars) + 2*np.random.normal(size=nobs)  # if H0 is true
else:
    y2 = 19 + np.dot(x2,[17.]*nvars) + 2*np.random.normal(size=nobs)

x2 = sm.add_constant(x2, prepend=False)

# stack
x = np.concatenate((x1,x2),0)
y = np.concatenate((y1,y2))
if example_groups == '2':
    groupind = (np.arange(2*nobs)>nobs-1).astype(int)
else:
    groupind = np.mod(np.arange(2*nobs),4)
    groupind.sort()
#x = np.column_stack((x,x*groupind[:,None]))


def print_results(res):
    groupind = res.groups
    #res.fitjoint()  #not really necessary, because called by ftest_summary
    ft = res.ftest_summary()
    #print ft[0]  #skip because table is nicer
    print('\nTable of F-tests for overall or pairwise equality of coefficients')
    # print 'hypothesis F-statistic         p-value  df_denom df_num  reject'
    for row in ft[1]:
        val = str(row)
        if row[1][1]<0.05:
            print(f'{val}*')
        else:
            print(val)
    from statsmodels.iolib import SimpleTable
    print(SimpleTable([([f'{row[0]!r}']
                        + list(row[1])
                        + ['*']*(row[1][1]>0.5).item() ) for row in ft[1]],
                      headers=['pair', 'F-statistic','p-value','df_denom',
                               'df_num']))

    print('Notes: p-values are not corrected for many tests')
    print('       (no Bonferroni correction)')
    print('       * : reject at 5% uncorrected confidence level')
    print('Null hypothesis: all or pairwise coefficient are the same')
    print('Alternative hypothesis: all coefficients are different')

    print('\nComparison with stats.f_oneway')
    print(stats.f_oneway(*[y[groupind==gr] for gr in res.unique]))
    print('\nLikelihood Ratio Test')
    print('likelihood ratio    p-value       df')
    print(res.lr_test())
    print('Null model: pooled all coefficients are the same across groups,')
    print('Alternative model: all coefficients are allowed to be different')
    print('not verified but looks close to f-test result')

    print('\nOLS parameters by group from individual, separate ols regressions')
    for group in sorted(res.olsbygroup):
        r = res.olsbygroup[group]
        print(group, r.params)

    print('\nCheck for heteroscedasticity, ')
    print('variance and standard deviation for individual regressions')
    print(' '*12, ' '.join('group %-10s' %(gr) for gr in res.unique))
    print('variance    ', res.sigmabygroup)
    print('standard dev', np.sqrt(res.sigmabygroup))

#now added to class
def print_results2(res):
    groupind = res.groups
    #res.fitjoint()  #not really necessary, because called by ftest_summary
    ft = res.ftest_summary()
    #print ft[0]  #skip because table is nicer
    templ = \
'''Table of F-tests for overall or pairwise equality of coefficients'
%(tab)s


Notes: p-values are not corrected for many tests
       (no Bonferroni correction)
       * : reject at 5%% uncorrected confidence level
Null hypothesis: all or pairwise coefficient are the same'
Alternative hypothesis: all coefficients are different'


Comparison with stats.f_oneway
%(statsfow)s


Likelihood Ratio Test
%(lrtest)s
Null model: pooled all coefficients are the same across groups,'
Alternative model: all coefficients are allowed to be different'
not verified but looks close to f-test result'


OLS parameters by group from individual, separate ols regressions'
%(olsbg)s
for group in sorted(res.olsbygroup):
    r = res.olsbygroup[group]
    print group, r.params


Check for heteroscedasticity, '
variance and standard deviation for individual regressions'
%(grh)s
variance    ', res.sigmabygroup
standard dev', np.sqrt(res.sigmabygroup)
'''

    from statsmodels.iolib import SimpleTable
    resvals = {}
    resvals['tab'] = str(SimpleTable([([f'{row[0]!r}']
                        + list(row[1])
                        + ['*']*(row[1][1]>0.5).item() ) for row in ft[1]],
                      headers=['pair', 'F-statistic','p-value','df_denom',
                               'df_num']))
    resvals['statsfow'] = str(stats.f_oneway(*[y[groupind==gr] for gr in
                                               res.unique]))
    #resvals['lrtest'] = str(res.lr_test())
    resvals['lrtest'] = str(SimpleTable([res.lr_test()],
                                headers=['likelihood ratio', 'p-value', 'df'] ))

    resvals['olsbg'] = str(SimpleTable([[group]
                                        + res.olsbygroup[group].params.tolist()
                                        for group in sorted(res.olsbygroup)]))
    resvals['grh'] = str(SimpleTable(np.vstack([res.sigmabygroup,
                                           np.sqrt(res.sigmabygroup)]),
                                 headers=res.unique.tolist()))

    return templ % resvals



#get results for example
#-----------------------

print('\nTest for equality of coefficients for all exogenous variables')
print('-------------------------------------------------------------')
res = OneWayLS(y,x, groups=groupind.astype(int))
print_results(res)

print('\n\nOne way ANOVA, constant is the only regressor')
print('---------------------------------------------')

print('this is the same as scipy.stats.f_oneway')
res = OneWayLS(y,np.ones(len(y)), groups=groupind)
print_results(res)


print('\n\nOne way ANOVA, constant is the only regressor with het is true')
print('--------------------------------------------------------------')

print('this is the similar to scipy.stats.f_oneway,')
print('but variance is not assumed to be the same across groups')
res = OneWayLS(y,np.ones(len(y)), groups=groupind.astype(str), het=True)
print_results(res)
print(res.print_summary()) #(res)
