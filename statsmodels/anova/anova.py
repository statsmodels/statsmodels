"""Analysis of variance (ANOVA)

Author: Yichuan Liu
"""
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from scipy import stats
import patsy
from statsmodels.iolib import summary2


def _ols_exclude_term(y, x, terms):
    """
    Linear regression excluding terms in x

    Parameters
    ----------
    y : array
        Dependent variable
    x : patsy.DesignMatrix
        Independent variables
    terms : a list of strings

    Returns
    -------
    residual sum of squares and degree of freedom

    """
    term_slices = x.design_info.term_name_slices
    ind = np.array([True]*x.shape[1])
    for term in terms:
        s = term_slices[term]
        ind[s] = False
    x = x[:, ind]
    model = OLS(y, x)
    results = model.fit()
    return (results.ssr, results.df_resid)


class ANOVA(object):
    """
    ANOVA using least square regression

    The full model regression residual sum of squared is
    used to compared with reduced model for calculating the
    within subject effect sum of squared

    Between subject effect is not yet supported

    Parameters
    ----------
    data : DataFrame
    dv : string
        Dependent variable
    within : a list of string(s)
        The within-subject factors
    between : a list of string(s)
        The between-subject factors
    subject : string
        Specify the subject id

    Returns
    -------
    ANOVAResults

    """

    def __init__(self, data, dv, within=None, between=None, subject=None):
        self.data = data
        self.dv = dv
        self.within = within
        self.between = between
        if between is not None:
            raise NotImplementedError('Between subject effect not '
                                      'yet supported!')
        self.subject = subject

    def fit(self):
        factors = self.within + [self.subject]
        factors = ['C(%s, Sum)' % i for i in factors]
        y = self.data[self.dv].values
        x = patsy.dmatrix('*'.join(factors), data=self.data)
        term_last = [':'.join(factors)]
        term_slices = x.design_info.term_name_slices
        ssr, df_resid = _ols_exclude_term(y, x, term_last)

        anova_table = pd.DataFrame(
            {'F Value':[], 'Num DF':[], 'Den DF':[], 'Pr > F':[]})
        for key in term_slices:
            if self.subject not in key and key != 'Intercept':
                ssr1, df_resid1 = _ols_exclude_term(
                    y, x, term_last + [key])
                df1 = df_resid1 - df_resid
                msm = (ssr1 - ssr) / df1
                if key == ':'.join(factors[:-1]):
                    msr = ssr / df_resid
                    df2 = df_resid
                else:
                    ssr1, df_resid1 = _ols_exclude_term(
                        y, x, term_last + [key +
                                           ':C(%s, Sum)' % self.subject])
                    df2 = df_resid1 - df_resid
                    msr = (ssr1 - ssr) / df2

                F = msm / msr
                p = stats.f.sf(F, df1, df2)
                term = key.replace('C(', '').replace(', Sum)', '')
                anova_table.loc[term, 'F Value'] = F
                anova_table.loc[term, 'Num DF'] = df1
                anova_table.loc[term, 'Den DF'] = df2
                anova_table.loc[term, 'Pr > F'] = p
        self.anova_table = anova_table.iloc[:, [1, 2, 0, 3]]
        return ANOVAResults(self)


class ANOVAResults(object):
    """
    ANOVA results class

    Attributes
    ----------
    anova_table : DataFrame
    """
    def __init__(self, anova):
        self.anova_table = anova.anova_table

    def __str__(self):
        return self.summary().__str__()

    def summary(self, contrast_L=False, transform_M=False):
        summ = summary2.Summary()
        summ.add_title('ANOVA')
        summ.add_df(self.anova_table)

        return summ


