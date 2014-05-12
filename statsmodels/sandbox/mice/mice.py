#import cython
import glob
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,"C:/Users/Frank/Documents/GitHub/statsmodels/")
import copy
#import statsmodels.api as sm
from statsmodels.regression import linear_model


class ImputedData:
    """
    Initialize a data object with missing data information and functionality 
    to insert values in missing data slots.
    """
    def __init__(self, data):
        self.data = pd.DataFrame(data)
        self.values = {}
        self.mean = {}
        for c in self.data.columns:
            temp = np.flatnonzero(pd.isnull(self.data[c]))
            self.values[c] = []
            self.values[c].append([])
            self.values[c].append([])
            self.values[c][0] = temp
            self.mean[c] = self.data[c].mean()

    def to_data_frame(self, copy=False):
       for k in self.values.keys():
           ix = self.values[k][0]
           v = self.values[k][1]
           self.data[k][ix] = v
       return self.data

    def to_array(self, copy=False):
        return np.asarray(self.to_data_frame(copy))

    def mean_fill(self):
        for c in self.data.columns:
            self.values[c][1] = self.mean[c]
        self.to_data_frame()

    def update_value(self, c, value):
        self.values[c][1] = np.asarray(value)

class Imputer:
    """
    Initializes object that imputes values for a single variable using a given formula
    """
    def __init__(self, data, formula, model_class, init_args={}, fit_args={}):
        self.data = data
        self.formula = formula
        self.model_class = model_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.endog_name = str(self.formula.split("~")[0].strip())

    # Impute the dependent variable once
    def impute_asymptotic_bayes(self):
        md = linear_model.OLS.from_formula(self.formula, self.data.data, **self.init_args)
        mdf = md.fit(**self.fit_args)

        self.exog_name = md.exog_names[1:]
        params = mdf.params.copy()
        covmat = mdf.cov_params()
        covmat_sqrt = np.linalg.cholesky(covmat)
        u = np.random.chisquare(mdf.df_resid)
        scale_per = mdf.mse_resid * mdf.df_resid/u
        p = len(params)
        params += np.dot(covmat_sqrt, np.random.normal(0,scale_per,p))
        ix = self.data.values[self.endog_name][0]
        exog = self.data.data[self.exog_name].ix[ix]
        new_endog = md.get_distribution(params=params, exog=exog, scale=scale_per).rvs()
        self.data.update_value(self.endog_name,new_endog)
        self.data.to_data_frame()

class ImputerChain:
    """
    Manage a collection of imputers for variables in a common dataframe. 
    This class does imputation and stores the imputed data sets, it does not fit
    the analysis model.
    """
    def __init__(self, imputer_list):
        self.imputer_list = imputer_list
        self.imputer_list[0].data.mean_fill()
        self.values = []
        self.implength = len(imputer_list)

    # Impute each variable once, initialize missing values to column means
    def cycle(self):
        for im in self.imputer_list:
            im.impute_asymptotic_bayes()

    # Impute data sets and save them to disk
    def generate_data(self, num, skip, base_name):
        for k in range(num):
            self.imputer_list[0].data.mean_fill()

            for j in range(skip):
                self.cycle()
            fname = "%s_%d.csv" % (base_name, k)
            self.imputer_list[0].data.data.to_csv(fname, index=False)
            self.values.append(copy.deepcopy(self.imputer_list[self.implength - 1].data.values))
            self.imputer_list[0].data.mean_fill()


class ImputerCombine:
    """
    Fits the analysis model to each imputed dataset and combines the results using Rubin's rule.
    """
    def __init__(self, imputer_chain, analysis_formula, analysis_class, iternum, cnum,
                 init_args={}, fit_args={}):
        imputer_chain.generate_data(iternum, cnum,'ftest')
        self.formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.iternum = iternum
        self.fname = glob.glob("*.csv")
        self.params_lst = []
        self.std_list = []

    def combine(self):
        for name in self.fname:
            dat = pd.read_csv(name)
            md = linear_model.OLS.from_formula(self.formula, dat, **self.init_args)
            mdf = md.fit(**self.fit_args)
            self.params_lst.append(mdf.params)
            self.std_list.append(mdf.bse)
        self.params = np.mean(self.params_lst, axis = 0)
        within_g = np.mean(self.std_list, axis = 0)
        between_g = np.std(self.params_lst, axis = 0)
        self.std = within_g + (1 + 1/self.iternum) * between_g
        
#not implemented yet
class AnalysisChain:
    
    def __init__(self, imputer_list, analysis_formula, analysis_class, iternum, cnum,
                 init_args={}, fit_args={}):
        self.imputer_list = imputer_list
        self.analysis_formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args

    def cycle(self):
        for im in self.imputer_list:
            im.impute()

    def run_chain(self, num, skip):
        params = []
        standard_errors = []
        for k in range(num):
            for j in range(skip):
                self.cycle()
                md = self.analysis_class.from_formula(
                            self.analysis_formula,
                             **self.init_args)
                mdf = md.fit(**self.fit_args)
                params.append(mdf.params)
                standard_errors.append(mdf.bse)

         ## apply the combining rule and return a results class