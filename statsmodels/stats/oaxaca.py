#TODO Variance can be calculated for the three_fold
#TODO Group Size Effects can be accounted for
#TODO Non-Linear Oaxaca-Blinder can be used
"""
Author: Austin Adams

This Class implements Oaxaca-Blinder Decomposition:

Two-Fold (two_fold)
Three-Fold (three_fold)

A Oaxaca-Blinder is a statistical method that is used to explain
the differences between two mean values. The idea is to show
from two mean values what can be explained by the data and 
what cannot by using OLS regression frameworks.

"The original use by Oaxaca's was to explain the wage 
differential between two different groups of workers, 
but the method has since been applied to numerous other topics." (Wikipedia)


The model is designed to accept two endogenous response variables
and two exogenous explanitory variables. They are then fit using
the specific type of decomposition that you want.


General reference for Oaxaca-Blinder:

B. Jann "The Blinder-Oaxaca decomposition for linear
regression models," The Stata Journal, 2008.

Econometrics references for regression models:

E. M. Kitagawa  "Components of a Difference Between Two Rates" 
Journal of the American Statistical Association, 1955.

A. S. Blinder "Wage Discrimination: Reduced Form and Structural
Estimates," The Journal of Human Resources, 1973.
"""
import statsmodels.api as sm
import numpy as np

class Oaxaca(object):
    """
    Class to perform Oaxaca-Blinder Decomposition.
    
    Parameters
    ----------
    endog: array-like
        'endog' is the endogenous variable or the dependent variable 
        that you are trying to explain.  
    exog: array-like
        'exog' is the exogenous variable(s) or the independent variable(s) 
        that you are using to explain the endogenous variable.
    bifurcate: int or string
        'bifurcate' is the column of the exogenous variable(s) that you 
        wish to split on. This would generally be the group that you wish
        to explain the two means for. Int of the column for a NumPy array 
        or int/string for the name of the column in Pandas
    hasconst: bool, optional
        Indicates whether the two exogenous variables include a user-supplied
        constant. If True, a constant is assumed. If False, a constant is added
        at the start. If nothing is supplied, then True is assumed.
    swap: bool, optional
        Imitates the STATA Oaxaca command by allowing users to choose to swap groups.
        Unlike STATA, this is assumed to be True instead of False
    cov_type: string, optional
        See regression.linear_model.RegressionResults for a description of the 
        available covariance estimators
    cov_kwdslist or None, optional
        See linear_model.RegressionResults.get_robustcov_results for a description 
        required keywords for alternative covariance estimators

    Attributes
    ----------
    None

    Methods:
    ----------
    three_fold()
        Returns the three-fold decomposition of Oaxaca-Blinder

    two_fold()
        Returns the two-fold decomposition of the Oaxaca-Blinder
        
    Notes
    -----
    Please check if your data includes at constant. This will still run, but
    will return extremely incorrect values if set incorrectly.

    You can access the models by using their code and the . syntax.
    _t_model for the total model, _f_model for the first model, 
    _s_model for the second model.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> import Oaxaca
    >>> data = sm.datasets.ccards.load()
    
    '3' is the column of which we want to explain or which indicates
    the two groups. In this case, it is if you rent.
    
    >>> model = Oaxaca(df.endog, df.exog, 3, hasconst = False)
    >>> model.two_fold()
    >>> ******************************
        Unexplained Effect: 27.94091
        Explained Effect: 130.80954
        Gap: 158.75044
        ******************************
    >>> model.three_fold()
    >>> ******************************
        Characteristic Effect: 321.74824
        Coefficent Effect: 75.45371
        Interaction Effect: -238.45151
        Gap: 158.75044
        ******************************
    """
    def __init__(self, endog, exog, bifurcate, hasconst = True, swap = True, cov_type = 'nonrobust', cov_kwds=None):
            if str(type(exog)).find('pandas') != -1:
                bifurcate = exog.columns.get_loc(bifurcate)
                endog, exog = np.array(endog), np.array(exog)

            bi_col = exog[:, bifurcate]
            endog = np.column_stack((bi_col, endog))
            bi = np.unique(bi_col)
            
            #split the data along the bifurcate axis, the issue is you need to delete it after you fit the model for the total model.
            exog_f = exog[np.where(exog[:, bifurcate] == bi[0])]
            exog_s = exog[np.where(exog[:, bifurcate] == bi[1])]
            endog_f = endog[np.where(endog[:, 0] == bi[0])]
            endog_s = endog[np.where(endog[:, 0] == bi[1])]
            exog_f = np.delete(exog_f, bifurcate, axis = 1)
            exog_s = np.delete(exog_s, bifurcate, axis = 1)
            endog_f = endog_f[:,1]
            endog_s = endog_s[:,1]
            endog = endog[:,1]
            
            self.gap = endog_f.mean() - endog_s.mean() 
            
            if swap and self.gap < 0:
                endog_f, endog_s = endog_s, endog_f
                exog_f, exog_s = exog_s, exog_f
                self.gap = endog_f.mean() - endog_s.mean()
            
            if hasconst == False:
                exog_f = sm.add_constant(exog_f, prepend = False)
                exog_s = sm.add_constant(exog_s, prepend = False)
                exog = sm.add_constant(exog, prepend = False)
            
            self._t_model = sm.OLS(endog, exog).fit(cov_type = cov_type, cov_kwds = cov_kwds)
            self._f_model = sm.OLS(endog_f, exog_f).fit(cov_type = cov_type, cov_kwds = cov_kwds)
            self._s_model = sm.OLS(endog_s, exog_s).fit(cov_type = cov_type, cov_kwds = cov_kwds)
            
            self.exog_f_mean = np.mean(exog_f, axis = 0)
            self.exog_s_mean = np.mean(exog_s, axis = 0)
            self.t_params = np.delete(self._t_model.params, bifurcate)
        
    def three_fold(self):
        """
        Calculates the three-fold Oaxaca Blinder Decompositions

        Parameters
        ----------

        None

        Returns
        -------

        char_eff : float
            This is the effect due to the group differences in
            predictors

        coef_eff: float
            This is the effect due to differences of the coefficients
            of the two groups
        
        int_eff: float
            This is the effect due to differences in both effects
            existing at the same time between the two groups.
        
        gap: float
            This is the gap in the mean differences of the two groups.
        """

        self.char_eff = (self.exog_f_mean - self.exog_s_mean) @ self._s_model.params
        self.coef_eff = (self.exog_s_mean) @ (self._f_model.params - self._s_model.params)
        self.int_eff = (self.exog_f_mean - self.exog_s_mean) @ (self._f_model.params - self._s_model.params)
        
        print("".join(["*" for x in range(0,30)]))
        print("Characteristic Effect: {:.5f}\nCoefficent Effect: {:.5f}\nInteraction Effect: {:.5f}\nGap: {:.5f}".format(self.char_eff, self.coef_eff, self.int_eff, self.gap))
        print("".join(["*" for x in range(0,30)]))
        
        return self.char_eff, self.coef_eff, self.int_eff, self.gap
    
    def two_fold(self):
        """
        Calculates the two-fold or pooled Oaxaca Blinder Decompositions

        Parameters
        ----------

        None

        Returns
        -------

        unexplained : float
            This is the effect that cannot be explained by the data at hand.
            This does not mean it cannot be explained with more.
        
        explained: float
            This is the effect that can be explained using the data.
        
        gap: float
            This is the gap in the mean differences of the two groups.
        """
        self.unexplained = (self.exog_f_mean @ (self._f_model.params - self.t_params)) + (self.exog_s_mean @ (self.t_params - self._s_model.params))
        self.explained = (self.exog_f_mean - self.exog_s_mean) @ self.t_params
        
        print("".join(["*" for x in range(0,30)]))
        print("Unexplained Effect: {:.5f}\nExplained Effect: {:.5f}\nGap: {:.5f}".format(self.unexplained, self.explained, self.gap))        
        print("".join(["*" for x in range(0,30)]))
        
        return self.unexplained, self.explained, self.gap