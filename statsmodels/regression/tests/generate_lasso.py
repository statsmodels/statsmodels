import numpy as np

"""
Generate data sets for testing linear_models fit_regularized

After running this script, rerun lasso_r_results.R in R to rebuild the
results file "glmnet_r_results.py".

Currently only tests OLS.  Our implementation covers GLS, but it's not
clear if glmnet does.
"""

ds_ix = 0

for n in 100,200,300:
    for p in 2,5,10,20:

        exog = np.random.normal(size=(n,p))
        params = (-1.)**np.arange(p)
        params[::3] = 0
        expval = np.dot(exog, params)
        endog = expval + np.random.normal(size=n)
        data = np.concatenate((endog[:,None], exog), axis=1)

        data -= data.mean(0)
        data /= data.std(0, ddof=1)

        fname = "results/lasso_data_%02d.csv" % ds_ix
        ds_ix += 1
        np.savetxt(fname, data, delimiter=",")
