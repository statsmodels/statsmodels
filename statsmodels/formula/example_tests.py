from statsmodels.datasets import longley
from statsmodels.formula.api import ols
from charlton.model_matrix import ModelMatrixColumnInfo
from charlton.api import model_spec_and_matrices

#from statsmodels.formula.eval import make_hypotheses_matrices

dta = longley.load_pandas()
formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
df = dta.data
#df.column_info = ModelMatrixColumnInfo(df.columns.tolist())
#spec, endog, exog = model_spec_and_matrices(formula, df)

results = ols(formula, df).fit()
code = '(GNPDEFL = GNP) & (UNEMP = 2) & (YEAR/1829 = 1)'
#R, Q = make_hypotheses_matrices(results, code)
#print R
#print Q
#f_test = results.f_test(R, Q)

#code = ' and '.join(dta.exog_name)
#R, Q = make_hypotheses_matrices(results, code)
#print R
#print Q
#f_test = results.f_test(R, Q)
#print f_test.fvalue
#print results.fvalue


from charlton.constraint import linear_constraint
exog_names = results.model.exog_names
LC = linear_constraint(code.replace("&", ","), exog_names)
