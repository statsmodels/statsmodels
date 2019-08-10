# TODO Non-Linear Regressions can be used
# TODO Further decomposition of the two_fold parameters i.e.
# the delta method for further two_fold detail
"""
Author: Austin Adams

This class implements Oaxaca-Blinder Decomposition. It returns
a OaxacaResults Class:

OaxacaBlinder:
Two-Fold (two_fold)
Three-Fold (three_fold)

OaxacaResults:
Table Summary (summary)

Oaxaca-Blinder is a statistical method that is used to explain
the differences between two mean values. The idea is to show
from two mean values what can be explained by the data and
what cannot by using OLS regression frameworks.

"The original use by Oaxaca's was to explain the wage
differential between two different groups of workers,
but the method has since been applied to numerous other
topics." (Wikipedia)

The model is designed to accept two endogenous response variables
and two exogenous explanitory variables. They are then fit using
the specific type of decomposition that you want.

The method was famously used in Card and Krueger's paper
"School Quality and Black-White Relative Earnings: A Direct Assessment" (1992)

General reference for Oaxaca-Blinder:

B. Jann "The Blinder-Oaxaca decomposition for linear
regression models," The Stata Journal, 2008.

Econometrics references for regression models:

E. M. Kitagawa  "Components of a Difference Between Two Rates"
Journal of the American Statistical Association, 1955.

A. S. Blinder "Wage Discrimination: Reduced Form and Structural
Estimates," The Journal of Human Resources, 1973.
"""
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import numpy as np
from textwrap import dedent


class OaxacaBlinder(object):
    """
    Class to perform Oaxaca-Blinder Decomposition.

    Parameters
    ----------
    endog : array_like
        The endogenous variable or the dependent variable that you are trying
        to explain.
    exog : array_like
        The exogenous variable(s) or the independent variable(s) that you are
        using to explain the endogenous variable.
    bifurcate : {int, str}
        The column of the exogenous variable(s) on which to split. This would
        generally be the group that you wish to explain the two means for.
        Int of the column for a NumPy array or int/string for the name of
        the column in Pandas.
    hasconst : bool, optional
        Indicates whether the two exogenous variables include a user-supplied
        constant. If True, a constant is assumed. If False, a constant is added
        at the start. If nothing is supplied, then True is assumed.
    swap : bool, optional
        Imitates the STATA Oaxaca command by allowing users to choose to swap
        groups. Unlike STATA, this is assumed to be True instead of False
    cov_type : str, optional
        See regression.linear_model.RegressionResults for a description of the
        available covariance estimators
    cov_kwds : dict, optional
        See linear_model.RegressionResults.get_robustcov_results for a
        description required keywords for alternative covariance estimators

    Notes
    -----
    Please check if your data includes at constant. This will still run, but
    will return incorrect values if set incorrectly.

    You can access the models by using their code as an attribute, e.g.,
    _t_model for the total model, _f_model for the first model, _s_model for
    the second model.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.ccards.load()

    '3' is the column of which we want to explain or which indicates
    the two groups. In this case, it is if you rent.

    >>> model = sm.OaxacaBlinder(df.endog, df.exog, 3, hasconst = False)
    >>> model.two_fold().summary()
    Oaxaca-Blinder Two-fold Effects
    Unexplained Effect: 27.94091
    Explained Effect: 130.80954
    Gap: 158.75044

    >>> model.three_fold().summary()
    Oaxaca-Blinder Three-fold Effects
    Endowments Effect: 321.74824
    Coefficient Effect: 75.45371
    Interaction Effect: -238.45151
    Gap: 158.75044
    """

    def __init__(self, endog, exog, bifurcate, hasconst=True,
                 swap=True, var=False, cov_type='nonrobust', cov_kwds=None):
        if str(type(exog)).find('pandas') != -1:
            bifurcate = exog.columns.get_loc(bifurcate)
            endog, exog = np.array(endog), np.array(exog)

        self.var = var
        self.bifurcate = bifurcate
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.neumark = np.delete(exog, bifurcate, axis=1)
        self.exog = exog
        bi_col = exog[:, bifurcate]
        endog = np.column_stack((bi_col, endog))
        bi = np.unique(bi_col)

        # split the data along the bifurcate axis, the issue is you need to
        # delete it after you fit the model for the total model.
        exog_f = exog[np.where(exog[:, bifurcate] == bi[0])]
        exog_s = exog[np.where(exog[:, bifurcate] == bi[1])]
        endog_f = endog[np.where(endog[:, 0] == bi[0])]
        endog_s = endog[np.where(endog[:, 0] == bi[1])]
        exog_f = np.delete(exog_f, bifurcate, axis=1)
        exog_s = np.delete(exog_s, bifurcate, axis=1)
        endog_f = endog_f[:, 1]
        endog_s = endog_s[:, 1]
        self.endog = endog[:, 1]

        self.len_f, self.len_s = len(endog_f), len(endog_s)
        self.gap = endog_f.mean() - endog_s.mean()

        if swap and self.gap < 0:
            endog_f, endog_s = endog_s, endog_f
            exog_f, exog_s = exog_s, exog_f
            self.gap = endog_f.mean() - endog_s.mean()

        if hasconst is False:
            exog_f = add_constant(exog_f, prepend=False)
            exog_s = add_constant(exog_s, prepend=False)
            self.exog = add_constant(self.exog, prepend=False)
            self.neumark = add_constant(self.neumark, prepend=False)

        self.exog_f_mean = np.mean(exog_f, axis=0)
        self.exog_s_mean = np.mean(exog_s, axis=0)

        if self.var is True:
            # instead of storing the exog arrays, we can store
            # the variance of the matricies
            # debate on if I should just calculate this all the time
            f_cov = exog_f - 1 * np.mean(exog_f)
            self.f_cov = (f_cov.T @ f_cov) / (len(f_cov) * (len(f_cov) - 1))
            s_cov = exog_s - 1 * np.mean(exog_s)
            self.s_cov = (s_cov.T @ s_cov) / (len(s_cov) * (len(s_cov) - 1))

        self._f_model = OLS(endog_f, exog_f).fit(
                                                cov_type=cov_type,
                                                cov_kwds=cov_kwds)
        self._s_model = OLS(endog_s, exog_s).fit(
                                                cov_type=cov_type,
                                                cov_kwds=cov_kwds)

    def variance(self, decomp_type):
        """
        A helper function to calculate the variance. Used to keep
        the decomposition functions cleaner
        """

        if decomp_type == 3:
            endow_var = ((
                        (self.exog_f_mean - self.exog_s_mean)
                        @ self._s_model.cov_params()
                        @ (self.exog_f_mean - self.exog_s_mean))
                        + (self._s_model.params @ (self.f_cov + self.s_cov)
                            @ self._s_model.params))

            coef_var = ((self.exog_s_mean @ (self._f_model.cov_params()
                        + self._s_model.cov_params()) @ self.exog_s_mean)
                        + ((self._f_model.params - self._s_model.params)
                        @ self.s_cov
                        @ (self._f_model.params - self._s_model.params)))

            int_var = (((self.exog_f_mean - self.exog_s_mean)
                        @ (self._f_model.cov_params()
                        + self._s_model.cov_params())
                        @ (self.exog_f_mean - self.exog_s_mean))
                        + (self._f_model.params - self._s_model.params)
                        @ (self.f_cov + self.s_cov)
                        @ (self._f_model.params - self._s_model.params))
            var = [endow_var, coef_var, int_var]
        '''
        if decomp_type == 2:
            explained_var = (((self.exog_f_mean - self.exog_s_mean)\
                            .reshape(-1,1)
                            @ np.array([((self.weight[0]
                            @ self._f_model.cov_params()
                            @ self.weight[0])
                            + (self.weight[1] @ self._s_model.cov_params()
                            @ self.weight[1]))])
                            @ (self.exog_f_mean - self.exog_s_mean))\
                            .reshape(-1,1)
                            + ((self.weight[0]
                            @ self._f_model.params + self.weight[1]
                            @ self._s_model.params)
                            @ (self.f_cov + self.s_cov)
                            @ (self.weight[0] @ self._f_model.params
                              + self.weight[1]
                            @ self._s_model.params)))
           unexplained_var = ((self.weight[1] @ self.exog_f_mean
                            + self.weight[0] @ self.exog_s_mean)
                            @ (self._f_model.cov_params()
                            + self._s_model.cov_params())
                            @ (self.weight[1] @ self.exog_f_mean
                            + self.weight[0] @ self.exog_s_mean)
                            + ((self._f_model.params - self._s_model.params)
                            @ (self.weight[1] @ self.f_cov @ self.weight[1]
                            + self.weight[0] @ self.s_cov @ self.weight[0])
                            @ (self._f_model.params - self._s_model.params)))
            var = [explained_var, unexplained_var]
        '''
        return(var)

    def three_fold(self):
        """
        Calculates the three-fold Oaxaca Blinder Decompositions

        Returns
        -------
        OaxacaResults
            A results container for the three-fold decomposition.
        """
        var_val = None
        self.endow_eff = (
                        (self.exog_f_mean - self.exog_s_mean)
                        @ self._s_model.params)
        self.coef_eff = self.exog_s_mean @ (self._f_model.params
                                            - self._s_model.params)
        self.int_eff = ((self.exog_f_mean - self.exog_s_mean)
                        @ (self._f_model.params - self._s_model.params))

        if self.var is True:
            var_val = self.variance(3)

        return OaxacaResults(
                            (self.endow_eff, self.coef_eff,
                                self.int_eff, self.gap), 3, var_val=var_val)

    def two_fold(self, two_fold_type='pooled', submitted_weight=None):
        """
        Calculates the two-fold or pooled Oaxaca Blinder Decompositions

        Methods
        -------
        two_fold_type: string, optional
            This method allows for the specific calculation of the
            non-discriminatory model. There are four different types
            available at this time. pooled, cotton, reimers, self_submitted.
            Pooled is assumed and if a non-viable parameter is given,
            pooled will be ran.

            pooled - This type assumes that the pooled model's parameters
            (a normal regression) is the non-discriminatory model.
            This includes the indicator variable. This is generally
            the best idea. If you have economic justification for
            using others, then use others.

            nuemark - This is similar to the pooled type, but the regression
            is not done including the indicator variable.

            cotton - This type uses the adjusted in Cotton (1988), which
            accounts for the undervaluation of one group causing the
            overevalution of another. It uses the sample size weights for
            a linear combination of the two model parameters

            reimers - This type uses a linear combination of the two
            models with both parameters being 50% of the
            non-discriminatory model.

            self_submitted - This allows the user to submit their
            own weights. Please be sure to put the weight of the larger mean
            group only. This should be submitted in the
            submitted_weights variable.

        submitted_weight: int/float, required only for self_submitted,
            This is the submitted weight for the larger mean. If the
            weight for the larger mean is p, then the weight for the
            other mean is 1-p. Only submit the first value.

        Returns
        -------
        OaxacaResults
            A results container for the two-fold decomposition.
        """
        var_val = None
        if two_fold_type == 'cotton':
            submitted_weight = [self.len_f / (self.len_f + self.len_s),
                                self.len_s / (self.len_f + self.len_s)]
            self.t_params = (
                            (self.len_f / (self.len_f + self.len_s)
                                * self._f_model.params)
                            + (self.len_s / (self.len_f + self.len_s)
                                * self._s_model.params))

        elif two_fold_type == 'reimers':
            submitted_weight = [.5, .5]
            self.t_params = .5 * (self._f_model.params + self._s_model.params)

        elif two_fold_type == 'self_submitted':
            if submitted_weight is None:
                raise ValueError('Please submit weights')
            submitted_weight = [submitted_weight, 1 - submitted_weight]
            self.t_params = (
                            submitted_weight[0] * self._f_model.params
                            + submitted_weight[1] * self._s_model.params)

        elif two_fold_type == 'nuemark':
            self._t_model = OLS(self.endog, self.neumark).fit(
                        cov_type=self.cov_type,
                        cov_kwds=self.cov_kwds)
            self.t_params = self._t_model.params
            submitted_weight = [self.t_params, 1 - self.t_params]

        else:
            self._t_model = OLS(self.endog, self.exog).fit(
                                    cov_type=self.cov_type,
                                    cov_kwds=self.cov_kwds)
            self.t_params = np.delete(self._t_model.params, self.bifurcate)
            submitted_weight = [self.t_params, 1 - self.t_params]

        self.weight = submitted_weight
        self.unexplained = ((self.exog_f_mean
                            @ (self._f_model.params - self.t_params))
                            + (self.exog_s_mean
                            @ (self.t_params - self._s_model.params)))
        self.explained = (self.exog_f_mean - self.exog_s_mean) @ self.t_params

        # if self.var == True:
        #     TODO
        #     var_val = self.variance(2)

        return OaxacaResults(
                            (self.unexplained, self.explained, self.gap),
                            2, var_val=var_val)


class OaxacaResults:
    """
    This class summarizes the fit of the OaxacaBlinder model.

    Use .summary() to get a table of the fitted values or
    use .params to receive a list of the values

    If a two-fold model was fitted, this will return
    unexplained effect, explained effect, and the
    mean gap. The list will be of the following order
    and type.

    unexplained : float
        This is the effect that cannot be explained by the data at hand.
        This does not mean it cannot be explained with more.
    explained : float
        This is the effect that can be explained using the data.
    gap : float
        This is the gap in the mean differences of the two groups.

    If a three-fold model was fitted, this will
    return characteristic effect, coefficient effect
    interaction effect, and the mean gap. The list will
    be of the following order and type.

    endowment effect : float
        This is the effect due to the group differences in
        predictors
    coefficient effect : float
        This is the effect due to differences of the coefficients
        of the two groups
    interaction effect : float
        This is the effect due to differences in both effects
        existing at the same time between the two groups.
    gap : float
        This is the gap in the mean differences of the two groups.

    Attributes
    ----------
    params
        A list of all values for the fitted models.
    """
    def __init__(self, results, model_type, var_val=None):
        self.params = results
        self.var = var_val
        self.model_type = model_type

    def summary(self):
        """
        Print a summary table with the Oaxaca-Blinder effects
        """
        if self.model_type == 2:
            print(dedent("""\
            Oaxaca-Blinder Two-fold Effects

            Unexplained Effect: {:.5f}
            Explained Effect: {:.5f}
            Gap: {:.5f}""".format(
                                self.params[0], self.params[1],
                                self.params[2])))

        if self.model_type == 3:
            print(dedent("""\
            Oaxaca-Blinder Three-fold Effects

            Endowment Effect: {:.5f}
            Coefficient Effect: {:.5f}
            Interaction Effect: {:.5f}
            Gap: {:.5f}""".format(
                            self.params[0], self.params[1],
                            self.params[2], self.params[3])))
