import pandas as pd
import numpy as np
import sys
sys.path.insert(0,"C:/Users/Frank/Documents/GitHub/statsmodels/")

from statsmodels.regression import linear_model
#import statsmodels as sm

class ImputedData:
    def __init__(self, data):
        self.data = pd.DataFrame(data)
        self.values = {}
        for c in self.data.columns:
            temp = np.flatnonzero(pd.isnull(self.data[c]))
            self.values[c] = []
            self.values[c].append([])
            self.values[c].append([])
            self.values[c][0] = temp

    def to_data_frame(self, copy=False):
       for k in self.values.keys():
           ix = self.values[k][0]
           v = self.values[k][1]
           self.data[k][ix] = v
       return self.data
    
    def to_array(self, copy=False):
        return np.asarray(self.to_data_frame(copy))

    def mean_fill(self):
        self.data = self.data.fillna(self.data.mean())
        for c in self.data.columns:
            self.values[c][1] = self.data[c].mean()
            
    def update_value(self, c, value):
        self.values[c][1] = np.asarray(value)
        
# Class defining imputation for one variable.
class Imputer:
    def __init__(self, data, formula, model_class, init_args={}, fit_args={}):        
        self.data = data
        self.formula = formula
        self.model_class = model_class
        self.init_args = init_args
        self.fit_args = fit_args        
        self.endog_name = str(self.formula.split("~")[0].strip())
        temp = str(self.formula.split("~")[1].strip())
        self.numexog = len(temp.split("+"))
        self.exog_name = []
        for i in range(0,self.numexog):
            self.exog_name.append(temp.split("+")[i].strip())

    # Impute the dependent variable once
    def impute_asymptotic_bayes(self):
        md = linear_model.OLS.from_formula(self.formula, self.data.data, **self.init_args)
        mdf = md.fit(**self.fit_args)
        params = mdf.params.copy()
        covmat = mdf.cov_params()
        covmat_sqrt = np.linalg.cholesky(covmat)
        u = np.random.chisquare(mdf.df_resid)
        scale_per = mdf.mse_resid * mdf.df_resid/u
        p = len(params)
        params += np.dot(covmat_sqrt, np.random.normal(0,scale_per,p))
        mdf.params = params
        ix = self.data.values[self.endog_name][0]
        exog = self.data.data[self.exog_name].ix[ix]
        new_endog = mdf.get_distribution(exog=exog, scale=scale_per)  
        self.data.update_value(self.endog_name,new_endog)
        self.data.to_data_frame()       

# Manage a collection of imputers for variables in a  common dataframe.  
#This class does imputation and stores the imputed data sets, it does not fit 
#the analysis model.
class ImputerChain:

    def __init__(self, imputer_list):
        self.imputer_list = imputer_list
        self.imputer_list[0].data.mean_fill()        

    # Impute each variable once, initialize missing values to column means
    def cycle(self):
        for im in self.imputer_list:
            im.impute_asymptotic_bayes() 
        self.data = im.data.to_data_frame()

    # Impute data sets and save them to disk
    def generate_data(self, num, skip, base_name):
        for k in range(num):
            for j in range(skip):
                self.cycle()
            fname = "%s_%d.csv" % (base_name, k)            
            self.data.to_csv(fname,index=False)

class ImputerCombine

class AnalysisChain:
    def __init__(self, imputer_list, analysis_formula,
                      analysis_class, init_args, fit_args):
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