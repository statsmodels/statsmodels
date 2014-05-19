import random
import operator
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "C:/Users/Frank/Documents/GitHub/statsmodels/")

class ImputedData:
    """
    Initialize a data object with missing data information and functionality
    to insert values in missing data slots.
    """
    def __init__(self, data):
        self.data = pd.DataFrame(data)
        self.columns = {}
        for c in self.data.columns:
            self.columns[c] = MissingDataInfo(self.data[c])
        self.data = self.data.fillna(self.data.mean())

    def store_changes(self, col=None):
        if col==None:
            for c in self.columns.keys():
                ix = self.columns[c].ix_miss
                v = self.values[c].values
                self.data[c].iloc[ix] = v
        else:
            ix = self.columns[col].ix_miss
            v = self.columns[col].values
            self.data[col].iloc[ix] = v

class Imputer:
    """
    Initializes object that imputes values for a single variable
    using a given formula.
    """
    def __init__(self, data, formula, model_class, init_args={}, fit_args={},
                 scale="fix", scale_value=None):
        self.data = data
        self.formula = formula
        self.model_class = model_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.endog_name = str(self.formula.split("~")[0].strip())
        self.num_missing = len(self.data.columns[self.endog_name].ix_miss)
        self.scale = scale
        self.scale_value = scale_value

    # Impute the dependent variable once
    def impute_asymptotic_bayes(self):
        io = self.data.columns[self.endog_name].ix_obs
        md = self.model_class.from_formula(self.formula, self.data.data.iloc[io,:], **self.init_args)
        mdf = md.fit(**self.fit_args)
        params = mdf.params.copy()
        covmat = mdf.cov_params()
        covmat_sqrt = np.linalg.cholesky(covmat)
        if self.scale == "fix":
            if self.scale_value is None:
                scale_per = 1
            else:
                scale_per = self.scale_value
        elif self.scale == "perturb_chi2":
            u = np.random.chisquare(mdf.df_resid)
            scale_per = scale_per = mdf.df_resid/u
        elif self.scale == "perturb_boot":
            pass
        p = len(params)
        params += np.dot(covmat_sqrt, np.random.normal(0, scale_per * mdf.scale, p))
        imiss = self.data.columns[self.endog_name].ix_miss
        exog_name = md.exog_names[1:]
        exog = self.data.data[exog_name].iloc[imiss,:]
        endog_obj = md.get_distribution(params=params, exog=exog, scale=scale_per * mdf.scale)
        new_endog = endog_obj.rvs()
        self.data.columns[self.endog_name].values = new_endog
        self.data.store_changes(self.endog_name)

    def impute_pmm(self, k0=1):
        io = self.data.columns[self.endog_name].ix_obs
        md = self.model_class.from_formula(self.formula, self.data.data.iloc[io,:], **self.init_args)
        mdf = md.fit(**self.fit_args)
        params = mdf.params.copy()
        covmat = mdf.cov_params()
        covmat_sqrt = np.linalg.cholesky(covmat)
        if self.scale == "fix":
            if self.scale_value is None:
                scale_per = 1
            else:
                scale_per = self.scale_value
        elif self.scale == "perturb_chi2":
            u = np.random.chisquare(mdf.df_resid)
            scale_per = scale_per = mdf.df_resid/u
        elif self.scale == "perturb_boot":
            pass
        p = len(params)
        params += np.dot(covmat_sqrt, np.random.normal(0, mdf.scale * scale_per, p))
        exog_name = md.exog_names[1:]
        exog = self.data.data[exog_name]
        exog.insert(0, 'Intercept', 1)
        endog_all = md.predict(params,exog)
        endog_matched = []
        imiss = self.data.columns[self.endog_name].ix_miss
        for mval in endog_all[imiss]:
            dist = abs(endog_all - mval)
            dist = sorted(range(len(dist)), key=lambda k: dist[k])
            endog_matched.append(random.choice(np.array(self.data.data[self.endog_name][dist[len(imiss):len(imiss) + k0]])))
#            else:
#                endog_matched.append(self.data.data[self.endog_name][dist[len(imiss):len(imiss) + k0]])

        new_endog = endog_matched
        self.data.columns[self.endog_name].values = new_endog
        self.data.store_changes(self.endog_name)

    def impute_bootstrap(self):
        pass

class ImputerChain:
    """
    Manage a collection of imputers for variables in a common dataframe.
    This class does imputation and stores the imputed data sets, it does not fit
    the analysis model.

    Note: All imputers must refer to the same data object
    """
    def __init__(self, imputer_list, iternum, skipnum):
        #The version of "data" that imputer modifies is outside the class.
        self.imputer_list = imputer_list
        #Impute variable with least missing observations first
        self.imputer_list.sort(key=operator.attrgetter('num_missing'))
        #All imputers must refer to the same data object
        self.inum = iternum
        self.snum = skipnum
        self.c = 0

    def __iter__(self):
        return self

    # Impute each variable once, initialize missing values to column means
    def next(self):
        if self.c >= self.inum:
            raise StopIteration
        for j in range(self.snum):
            for im in self.imputer_list:
                im.impute_asymptotic_bayes()
        self.c = self.c + 1
        print self.c
        return self.imputer_list[0].data.data

    # Impute data sets and save them to disk, keep this around for now
#    def generate_data(self, num, skip, base_name):
#        for k in range(num):
#            for j in range(skip):
#                self.next()
#            fname = "%s_%d.csv" % (base_name, k)
#            self.imputer_list[0].data.data.to_csv(fname, index=False)
#            self.values.append(copy.deepcopy(self.imputer_list[self.implength - 1].data.values))
#            #self.imputer_list[0].data.mean_fill()


class MICE:
    """
    Fits the analysis model to each imputed dataset and combines the
    results using Rubin's rule.
    """
    def __init__(self, imputer_chain, analysis_formula, analysis_class,
                 init_args={}, fit_args={}):
        self.imputer_chain = imputer_chain
        self.formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.iternum = imputer_chain.inum

    def combine(self):
        params_list = []
        cov_list = []
        for data in self.imputer_chain:
            md = self.analysis_class.from_formula(self.formula, data, **self.init_args)
            mdf = md.fit(**self.fit_args)
            params_list.append(mdf.params)
            cov_list.append(np.array(mdf.normalized_cov_params))
        params = np.mean(params_list, axis=0)
        within_g = np.mean(cov_list, axis=0)
        between_g = np.cov(np.array(params_list).T, bias=1)
        cov_params = within_g + (1 + 1/self.iternum) * between_g
        #TODO: return results class, stuffed into mdf for now
        mdf._results.params = params
        mdf._results.normalized_cov_params = cov_params
        return mdf

class AnalysisChain:
    """
    Imputes and fits analysis model without saving intermediate datasets.
    """

    def __init__(self, imputer_list, analysis_formula, analysis_class,
                 init_args={}, fit_args={}):
        self.imputer_list = imputer_list
        #impute variable with least missing observations first
        self.imputer_list.sort(key=operator.attrgetter('num_missing'))
        self.analysis_formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args

    def cycle(self):
        for im in self.imputer_list:
            im.impute_pmm(5)

    def run_chain(self, num, skip):
        params_list = []
        cov_list = []
        for k in range(num):
            for j in range(skip):
                self.cycle()
            md = self.analysis_class.from_formula(self.analysis_formula, self.imputer_list[0].data.data, **self.init_args)
            mdf = md.fit(**self.fit_args)
            params_list.append(mdf.params)
            cov_list.append(np.array(mdf.normalized_cov_params))
        params = np.mean(params_list, axis=0)
        within_g = np.mean(cov_list, axis=0)
        between_g = np.cov(np.array(params_list).T, bias=1)
        cov_params = within_g + (1 + 1/num) * between_g
        #TODO: return results class, stuffed into mdf for now
        mdf._results.params = params
        mdf._results.normalized_cov_params = cov_params
        return mdf

class MissingDataInfo:

    def __init__(self, data):
        null = pd.isnull(data)
        self.ix_miss = np.flatnonzero(null)
        self.ix_obs = np.flatnonzero(~null)
        if len(self.ix_obs) == 0:
            raise ValueError("Variable to be imputed has no observed values")
        self.values = np.zeros(len(self.ix_miss), dtype=data.dtype)