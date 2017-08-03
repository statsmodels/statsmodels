from __future__ import division
import numpy as np
import pandas as pd
import summary_stats as ss

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

    def _stderr(self):
        # Essentially, we are calculating a total for each level combination
        # between the two variables. Using pandas doesnt allow for the use of
        # summary stats to compute the linearized stderr. Thus, this function
        # gets the indices that make up each 'group', and calculates the stderr
        # however, the indices are a dictionary, so we cant currently match the
        # stderr to the total calculated for each group
        for ind in self.df_group.indices.values():
            # make vector of zeros
            group_weights = np.zeros(self._m)
            # except at the indices in a particular group
            group_weights[ind] = self.design.weights[ind]
            group_design = ss.SurveyDesign(self.design.strat, self.design.clust,
                                           group_weights)
            stderr = ss.SurveyTotal(group_design, np.ones(self._m),
                                    cov_method='linearized', center_by='stratum').stderr


"""
issues:
    - currently, there is no way to know which stderr corresponds to
    which combination of levels between the two variables
    - I could do this if I incorporated SurveyDesign, bc each total
    automotically comes wht the stderr. However, not sure how to keep it in
    a table format, as it can not consider each combination between possible
    factors if it is not in the data (ie we can't do a 'fill_value=0' to show
    that certain combinations are not in the data)