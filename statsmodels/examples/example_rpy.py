'''Just two examples for using rpy

These examples are mainly for developers.

# example 1: OLS using LM
# example 2: GLM with binomial family
    The second results isn't exactly correct since it assumes that each
    obvervation has the same number of trials see datasets/longley for an R script
    with the correct syntax.

See rmodelwrap.py in the tests folder for a convenience wrapper
to make rpy more like statsmodels.  Note, however, that rmodelwrap
was created in a very ad hoc manner and due to the idiosyncracies in R
it does not work for all types of R models.

There are also R scripts included with most of the datasets to run
some basic models for comparisons of results to statsmodels.
'''

from __future__ import print_function
from statsmodels.compat.python import iterkeys
from rpy import r
import numpy as np
import statsmodels.api as sm


examples = [1, 2]

if 1 in examples:
    data = sm.datasets.longley.load()
    y,x = data.endog, sm.add_constant(data.exog, prepend=False)
    des_cols = ['x.%d' % (i+1) for i in range(x.shape[1])]
    formula = r('y~%s-1' % '+'.join(des_cols))
    frame = r.data_frame(y=y, x=x)
    results = r.lm(formula, data=frame)
    print(list(iterkeys(results)))
    print(results['coefficients'])

if 2 in examples:
    data2 = sm.datasets.star98.load()
    y2,x2 = data2.endog, sm.add_constant(data2.exog, prepend=False)
    import rpy
    y2 = y2[:,0]/y2.sum(axis=1)
    des_cols2 = ['x.%d' % (i+1) for i in range(x2.shape[1])]
    formula2 = r('y~%s-1' % '+'.join(des_cols2))
    frame2 = r.data_frame(y=y2, x=x2)
    results2 = r.glm(formula2, data=frame2, family='binomial')
    params_est = [results2['coefficients'][k] for k
                    in sorted(results2['coefficients'])]
    print(params_est)
    print(', '.join(['%13.10f']*21) % tuple(params_est))

