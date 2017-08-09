from __future__ import division
import numpy as np
import pandas as pd
import summary_stats as ss
import statsmodels.api as sm
from patsy.contrasts import Treatment
from patsy.contrasts import Poly

# TODO: compute corrected statistics
# add documentation
# start testing
class SurveyTable(object):
    # assumes data is nx2 ie the user specified which two
    # columns to work with
    def __init__(self, design, data):
        self.design = design
        # i could also treat the levels of the two vars as nested
        # ie put them in replacement for 'strata' and 'cluster' in surveydesign.
        # the benefits is that it's easier to get the totals, SE, etc
        # but it'll not be as intuitive to derive col_prop, cell_prop, etc
        self._m, self._p =  data.shape
        self.df = pd.concat([pd.DataFrame(data), pd.DataFrame(weights)], axis=1)

        # if just given one column
        if self.df.shape[1] == 2:
            self.df.columns = ['var1', 'weights']
            self.df_group = self.df.groupby(['var1'])
            self.table = self.df_group.weights.sum().unstack(fill_value=0)
        elif self.df.shape[1] == 3:
            self.df.columns = ["var1", "var2", "weights"]
            self.df_group = self.df.groupby(['var1', 'var2'])

            # the 'total' for each group
            self.table = self.df_group.weights.sum().unstack(fill_value=0)

        else:
            return ValueError("data should only have 1 or 2 columns")

        self._row_sum = self.table.sum(axis=1) # shape is (var1_unique_labs, )
        self._col_sum = self.table.sum(axis=0) # shape is (var2_unique_labs, )
        self._row_prop = self.table.div(self._row_sum, axis=0)
        self._col_prop = self.table.div(self._col_sum, axis=1)
        self._tot_sum = self.table.sum().sum()
        self._cell_prop = self.table / self._tot_sum
        self._row_marginal = self._row_sum / self._tot_sum
        self._col_marginal = self._col_sum / self._tot_sum
        # estimated proportion under the null hypothesis of independence
        self._null = np.outer(self._row_marginal, self._col_marginal)



    def __str__(self):
        tab = self._row_prop.copy()
        tab['col_tot'] = tab.sum(axis=1)
        # tab.loc['row_tot'] = tab.sum(axis=0)
        print(tab)
        return 'cell_proportions'

    def test_pearson(self):
        cell_diff_square = np.square((self._cell_prop - self._null))
        # uncorrected stat
        self.pearson = self._m * (cell_diff_square / self._null).sum().sum()

    def test_lrt(self):
        # Note: this is not definited if there are zeros in self.table
        if 0 in self.table:
            raise ValueError("table should not contain 0 for test_lrt")
        # uncorrected stat
        self.lrt = (self._col_prop * np.log(self._col_prop / self._null)).sum().sum()
        self.lrt = 2 * self._m

    def _stderr(self, cell_prop=True):
        # Essentially, we are calculating a total for each level combination
        # between the two variables. Using pandas doesnt allow for the use of
        # summary stats to compute the linearized stderr. Thus, this function
        # gets the indices that make up each 'group', and calculates the stderr
        # however, the indices are a dictionary, so we cant currently match the
        # stderr to the total calculated for each group
        self.stderr_dict = {}
        for ind in self.df_group.indices.values():
            # make vector of zeros
            group_weights = np.zeros(self._m)
            # except at the indices in a particular group
            group_weights[ind] = self.design.weights[ind]
            # if cell_prop is true, we calculate the variance of p_rc under the survey design
            if cell_prop:
                group_weights[ind] /= self.design.weights.sum()
            group_design = ss.SurveyDesign(self.design.strat, self.design.clust,
                                           group_weights)
            self.stderr_dict[tuple(ind)] = ss.SurveyTotal(group_design, np.ones(self._m),
                                    cov_method='linearized', center_by='stratum').stderr

    def _delta(self):
        D_inv = np.linarg.inv(np.diag(st._cell_prop.values))
        v_hat = np.diag(self._stderr())
        # need to get off diagonal elements of v_srs. But can't find what
        # 'p_st' is in the documentation
        v_srs = np.diag(self._cell_prop * (1 - self._cell_prop) / self._m)
        dat = pd.DataFrame(self.df.var1.astype('category').cat.codes,
                           self.df.var2.astype('category').cat.codes)
        dat = sm.add_constant(dat)
        dat.columns = ['const', 'var1', 'var2']
        model_fit = sm.OLS(dat[['const', 'var1']], dat['var2']).fit()
        # Does not fit the dimensions RCx(R-1)(C-1)
        # R is the number of rows in the table, C is number of columns
        model_fit.resid
