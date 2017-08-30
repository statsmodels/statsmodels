from __future__ import division
import numpy as np
import pandas as pd
from statsmodels.survey import summary_stats as ss


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
        self.df = pd.concat([pd.DataFrame(data), pd.DataFrame(self.design.weights)], axis=1)

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

    def test_pearson(self, cell_prop=True):
        cell_diff_square = np.square((self._cell_prop - self._null))
        # uncorrected stat
        self.pearson = self._m * (cell_diff_square / self._null).sum().sum()
        # rao and scott correction
        self._delta(cell_prop)
        if self._delta_est.ndim == 1:
            self._trace = self.delta_est
            self._trace_sq = np.square(self._delta_est)
        else:
            self._trace = np.trace(self._delta_est)
            self._trace_sq = np.trace(np.square(self._delta_est))

        self.pearson *= (self._trace / self._trace_sq)
        self.dof = np.square(self._trace) / self._trace_sq

    def test_lrt(self, cell_prop=True):
        # Note: this is not definited if there are zeros in self.table
        if 0 in self.table:
            raise ValueError("table should not contain 0 for test_lrt")
        # uncorrected stat
        self.lrt = (self._col_prop * np.log(self._col_prop / self._null)).sum().sum()
        self.lrt = 2 * self._m
        self._delta(cell_prop)
        if self._delta_est.ndim == 1:
            self._trace = self.delta_est
            self._trace_sq = np.square(self._delta_est)
        else:
            self._trace = np.trace(self._delta_est)
            self._trace_sq = np.trace(np.square(self._delta_est))
        self.lrt *= (self._trace / self._trace_sq)
        dof = np.square(self._trace) / self._trace_sq

    def _group_variance(self, cell_prop=True):
        # Essentially, we are calculating a total for each level combination
        # between the two variables. Using pandas doesnt allow for the use of
        # summary stats to compute the linearized stderr. Thus, this function
        # gets the indices that make up each 'group', and calculates the stderr
        # however, the indices are a dictionary, so we cant currently match the
        # stderr to the total calculated for each group
        self.var_dict = {}
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
            self.var_dict[tuple(ind)] = np.square(ss.SurveyTotal(group_design, np.ones(self._m),
                                    cov_method='linearized', center_by='stratum').stderr)
            group_var = np.asarray(list(self.var_dict.values()))
            group_var = group_var.reshape(len(group_var), )
            self.group_var = group_var
        return group_var

    def _delta(self, cell_prop=True):
        D_inv = np.linalg.inv(np.diag(self._cell_prop.values.flatten()))
        # this may or may not be in order bc _group_variance() used dict
        # indices to grab the observations, and dictionaries aren't ordered
        v_hat = np.diag(self._group_variance())

        # get v_srs using cell proportions or proportions under the null
        if cell_prop:
            v_srs = np.outer(self._col_prop, self._col_prop) / self._m
        else:
            v_srs = np.outer(self._null, self._null) / self._m

        b = self._contrast_matrix()
        self.b = b.copy()


        delta_numer = np.dot(b.T, D_inv).dot(v_hat).dot(D_inv).dot(b)
        delta_denom = np.linalg.inv(np.dot(b.T, D_inv).dot(v_srs).dot(D_inv).dot(b))
        self._delta_est = np.dot(delta_denom, delta_numer)

    def _contrast_matrix(self):
        R, C = self.table.shape

        ir = np.zeros((R, C))
        ir[0, :] = 1
        ir = ir.ravel()

        mat = []
        for i in range(R):
            mat.append(ir)
            ir = np.roll(ir, C)

        ic = np.zeros((R, C))
        ic[:, 0] = 1
        ic = ic.ravel()

        for i in range(C):
            mat.append(ic)
            ic = np.roll(ic, R)

        mat = np.asarray(mat).T
        self.mat = mat.copy()
        u,s,vt = np.linalg.svd(mat, 0)
        b = u[:, s>1e-12]

        qm = np.eye(b.shape[0]) - np.dot(b, b.T)
        u,s,vt = np.linalg.svd(qm, 0)
        b = u[:, s > 1e-12]
        return b

""""
questions:
- getting travis to be successful
- says 'no module names summary_stats', etc
- have kerby look at SAS code

"""