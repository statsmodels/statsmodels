"""Example: statsmodels.sysreg.sysmodel
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.sysreg.sysmodel import *

# This example uses the subset of the Grunfeld data in Greene's Econometric
# Analysis Chapter 14 (5th Edition)

grun_data = sm.datasets.grunfeld.load()

# Lexical order as R
firms = ['General Motors', 'Chrysler', 'General Electric', 'Westinghouse',
        'US Steel']
firms.sort()

grun_exog = grun_data.exog
grun_endog = grun_data.endog

# Construct the system equation by equation
sys = []
for f in firms:
    eq_f = {}
    index_f = grun_exog['firm'] == f
    eq_f['endog'] = grun_endog[index_f]
    exog = (grun_exog[index_f][var] for var in ['value', 'capital'])
    eq_f['exog'] = np.column_stack(exog)
    eq_f['exog'] = sm.add_constant(eq_f['exog'], prepend=True)
    sys.append(eq_f)

# SUR estimation
grun_mod = SysSUR(sys)
grun_res = grun_mod.fit()

