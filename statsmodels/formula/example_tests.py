from statsmodels.datasets import longley
from statsmodels.formula.api import ols
from statsmodels.formula.eval import make_hypotheses_matrices

dta = longley.load_pandas()
formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
df = dta.data
results = ols(formula, df).fit()
code = '(GNPDEFL = GNP) & (UNEMP = 2) & (YEAR/1829 = 1)'
R, Q = make_hypotheses_matrices(results, code)
print R
print Q
f_test = results.f_test(R, Q)

code = ' and '.join(dta.exog_name)
R, Q = make_hypotheses_matrices(results, code)
print R
print Q
f_test = results.f_test(R, Q)
print f_test.fvalue
print results.fvalue
