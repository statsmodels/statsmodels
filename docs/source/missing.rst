:orphan:

.. _missing_data:

Missing Data
------------
All of the models can handle missing data. For performance reasons, the default is not to do any checking for missing data. If, however, you would like for missing data to be handled internally, you can do so by using the missing keyword argument. The default is to do nothing

.. ipython:: python

   import statsmodels.api as sm
   data = sm.datasets.longley.load(as_pandas=False)
   data.exog = sm.add_constant(data.exog)
   # add in some missing data
   missing_idx = np.array([False] * len(data.endog))
   missing_idx[[4, 10, 15]] = True
   data.endog[missing_idx] = np.nan
   ols_model = sm.OLS(data.endog, data.exog)
   ols_fit = ols_model.fit()
   print(ols_fit.params)

This silently fails and all of the model parameters are NaN, which is probably not what you expected. If you are not sure whether or not you have missing data you can use `missing = 'raise'`. This will raise a `MissingDataError` during model instantiation if missing data is present so that you know something was wrong in your input data.

.. ipython:: python
   :okexcept:

   ols_model = sm.OLS(data.endog, data.exog, missing='raise')

If you want statsmodels to handle the missing data by dropping the observations, use `missing = 'drop'`.

.. ipython:: python

   ols_model = sm.OLS(data.endog, data.exog, missing='drop')

We are considering adding a configuration framework so that you can set the option with a global setting.
