"""Analysis of variance (ANOVA)

Author: Yichuan Liu
"""
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from scipy import stats
import patsy
from statsmodels.iolib import summary2


class ANOVA(object):
    """
    ANOVA using least square regression

    The full model regression residual sum of squared is
    used to compared with reduced model for calculating the
    within subject effect sum of squared [1]

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

    .. [1] Rutherford, Andrew. ANOVA and ANCOVA: a GLM approach. John Wiley & Sons, 2011.

    """

    def __init__(self, data, dv, within=None, between=None, subject=None):
        self.data = data
        self.dv = dv
        self.within = within
        for factor in within:
            if factor == 'C':
                raise ValueError('Factor name cannot be "C"! This is '
                                 'conflict with patsy contrast function '
                                 'name')
        self.between = between
        if between is not None:
            raise NotImplementedError('Between subject effect not '
                                      'yet supported!')
        self.subject = subject

    def fit(self):
        y = self.data[self.dv].values

        def _not_slice(slices, slices_to_exclude, n):
            ind = np.array([True]*n)
            for term in slices_to_exclude:
                s = slices[term]
                ind[s] = False
            return ind

        def _ssr_reduced_model(y, x, term_slices, params, keys):
            ind = _not_slice(term_slices, keys, x.shape[1])
            params1 = params[ind]
            ssr = np.subtract(y, x[:, ind].dot(params1))
            ssr = ssr.T.dot(ssr)
            df_resid = len(y) - len(params1)
            return ssr, df_resid

        within = ['C(%s, Sum)' % i for i in self.within]
        subject = 'C(%s, Sum)' % self.subject
        factors = within + [subject]
        x = patsy.dmatrix('*'.join(factors), data=self.data)
        term_slices = x.design_info.term_name_slices
        for key in term_slices:
            ind = np.array([False]*x.shape[1])
            ind[term_slices[key]] = True
            term_slices[key] = np.array(ind)
        term_exclude = [':'.join(factors)]
        ind = _not_slice(term_slices, term_exclude, x.shape[1])
        x = x[:, ind]
        model = OLS(y, x)
        results = model.fit()
        for i in term_exclude:
            term_slices.pop(i)
        for key in term_slices:
            term_slices[key] = term_slices[key][ind]
        params = results.params
        df_resid = results.df_resid
        ssr = results.ssr

        anova_table = pd.DataFrame(
            {'F Value':[], 'Num DF':[], 'Den DF':[], 'Pr > F':[]})

        for key in term_slices:
            if self.subject not in key and key != 'Intercept':
                #  Independen variables are orthogonal
                ssr1, df_resid1 = _ssr_reduced_model(
                    y, x, term_slices, params, [key])
                df1 = df_resid1 - df_resid
                msm = (ssr1 - ssr) / df1
                if key == ':'.join(factors[:-1]) or (key + ':' +
                                                     subject not in term_slices):
                    mse = ssr / df_resid
                    df2 = df_resid
                else:
                    ssr1, df_resid1 = _ssr_reduced_model(
                        y, x, term_slices, params,
                        [key+ ':' + subject])
                    df2 = df_resid1 - df_resid
                    mse = (ssr1 - ssr) / df2
                F = msm / mse
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


