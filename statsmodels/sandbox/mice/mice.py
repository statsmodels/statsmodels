import random
import operator
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "C:/Users/Frank/Documents/GitHub/statsmodels/")
import statsmodels.api as sm

class ImputedData(object):
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

    def new_imputer(self, endog, formula=None, model_class=None, init_args={}, fit_args={}, scale="fix", scale_value=None):
        """
        Create Imputer instance from our ImputedData instance
        """
        if model_class is None:
            model_class = sm.OLS
        if formula is None:
            #check this
            default_formula = endog + " ~ " + " + ".join([x for x in self.data.columns if x != endog])
            return Imputer(default_formula, model_class, self, init_args=init_args, fit_args=fit_args, scale=scale,scale_value=scale_value)
        else:
            formula = endog + " ~ " + formula
            return Imputer(formula, model_class, self, init_args=init_args, fit_args=fit_args, scale=scale,scale_value=scale_value)

    def store_changes(self, vals, col=None):
        """
        Fill in dataset with imputed values
        """
        if col==None:
            for c in self.columns.keys():
                ix = self.columns[c].ix_miss
                self.data[c].iloc[ix] = vals
        else:
            ix = self.columns[col].ix_miss
            self.data[col].iloc[ix] = vals

class Imputer(object):
    """
    Initializes object that imputes values for a single variable
    using a given formula.
    """
    def __init__(self, formula, model_class, data, init_args={}, fit_args={},
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

    def impute_asymptotic_bayes(self):
        """
        Use Gaussian approximation to posterior distribution to simulate data
        """
        io = self.data.columns[self.endog_name].ix_obs
        md = self.model_class.from_formula(self.formula, self.data.data.iloc[io,:], **self.init_args)
        mdf = md.fit(**self.fit_args)
        params = mdf.params.copy()
        covmat = mdf.cov_params()
        covmat_sqrt = np.linalg.cholesky(covmat)
        if self.scale == "fix":
            if self.scale_value is None:
                scale_per = 1.
            else:
                scale_per = self.scale_value
        elif self.scale == "perturb_chi2":
            u = np.random.chisquare(mdf.df_resid)
            scale_per = mdf.df_resid/u
        elif self.scale == "perturb_boot":
            pass
        p = len(params)
        params += np.dot(covmat_sqrt, np.random.normal(0, scale_per * mdf.scale, p))
        imiss = self.data.columns[self.endog_name].ix_miss
        #find a better way to determine if first column is intercept
        exog_name = md.exog_names[1:]
        exog = self.data.data[exog_name].iloc[imiss,:]
        endog_obj = md.get_distribution(params=params, exog=exog, scale=scale_per * mdf.scale)
        new_endog = endog_obj.rvs()
        self.data.store_changes(new_endog, self.endog_name)

    def impute_pmm(self, k0=1):
        """
        Use predictive mean matching to simulate data
        """
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
        new_endog = endog_matched
        self.data.store_changes(new_endog, self.endog_name)

    def impute_bootstrap(self):
        pass

#TODO: put imputer type, optional params into this class
class ImputerChain(object):
    """
    Manage a collection of imputers for variables in a common dataframe.
    This class does imputation and stores the imputed data sets, it does not fit
    the analysis model.

    Note: All imputers must refer to the same data object
    """
    def __init__(self, imputer_list):
        self.imputer_list = imputer_list
        #Impute variable with least missing observations first
        self.imputer_list.sort(key=operator.attrgetter('num_missing'))
        #All imputers must refer to the same data object
        self.data = imputer_list[0].data.data


    def __iter__(self):
        return self

    # Impute each variable once
    def next(self):
        """
        Make this class an iterator that returns imputed datasets after each imputation.
        Not all returned datsets are saved!
        """
        for im in self.imputer_list:
            im.impute_pmm()
        return self.data

    # Impute data sets and save them to disk, keep this around for now
#    def generate_data(self, num, skip, base_name):
#        for k in range(num):
#            for j in range(skip):
#                self.next()
#            fname = "%s_%d.csv" % (base_name, k)
#            self.imputer_list[0].data.data.to_csv(fname, index=False)
#            self.values.append(copy.deepcopy(self.imputer_list[self.implength - 1].data.values))
#            #self.imputer_list[0].data.mean_fill()

class AnalysisChain(object):
    """
    Fits the model of analytical interest to each dataset.
    Datasets are chosen after an initial burnin period and also after
    skipping a set number of imputations for each iteration.
    """

    def __init__(self, imputer_chain, analysis_formula, analysis_class, skipnum,
                 burnin, init_args={}, fit_args={}):
        self.imputer_chain = imputer_chain
        self.analysis_formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.skipnum = skipnum
        self.burnin = burnin
        self.burned = True


    def __iter__(self):
        return self

    def next(self):
        """
        Makes this class an iterator that returns the fitted analysis model.
        """
        scount = 0
        while scount < self.skipnum:
            if self.burned:
                for b in range(self.burnin):
                    self.imputer_chain.next()
                self.burned = False
            else:
                scount += 1
                if scount == self.skipnum:
                    data = self.imputer_chain.next()
                else:
                    self.imputer_chain.next()
            print scount
        md = self.analysis_class.from_formula(self.analysis_formula, data, **self.init_args)
        mdf = md.fit(**self.fit_args)
        return mdf

class MICE(object):
    """
    Fits the analysis model to each imputed dataset and combines the
    results using Rubin's rule.
    """
    def __init__(self, analysis_formula, analysis_class, imputer_list,
                 init_args={}, fit_args={}):
        self.imputer_list = imputer_list
        self.analysis_formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args

    def combine(self, iternum, skipnum, burnin=5):
        """
        Combines model results and returns the model of itnerest with pooled estimates/covariance matrix
        """
        imp_chain = ImputerChain(self.imputer_list)
        analysis_chain = AnalysisChain(imp_chain, self.analysis_formula, self.analysis_class, skipnum, burnin,
                                       self.init_args, self.fit_args)
        params_list = []
        cov_list = []
        scale_list = []
        current_iter = 0
        while current_iter < iternum:
            model = analysis_chain.next()
            params_list.append(model.params)
            cov_list.append(np.array(model.normalized_cov_params))
            scale_list.append(model.scale)
            current_iter += 1
            if current_iter == iternum:
                md = model
            print current_iter
        scale = np.mean(scale_list)
        params = np.mean(params_list, axis=0)
        within_g = np.mean(cov_list, axis=0)
        between_g = np.cov(np.array(params_list).T, bias=1)
        cov_params = within_g + (1 + 1/float(iternum)) * between_g
        md._results.params = params
        md._results.scale = scale
        md._results.normalized_cov_params = cov_params
        return md

class MissingDataInfo(object):
    """
    Contains all the missing data information from the passed-in data object. One for each column!
    """

    def __init__(self, data):
        null = pd.isnull(data)
        self.ix_miss = np.flatnonzero(null)
        self.ix_obs = np.flatnonzero(~null)
        if len(self.ix_obs) == 0:
            raise ValueError("Variable to be imputed has no observed values")