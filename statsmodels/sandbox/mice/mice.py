import pandas as pd
import numpy as np
import sys
sys.path.insert(0,"C:/Users/Frank/Documents/GitHub/statsmodels/")

from statsmodels.regression import linear_model
#import statsmodels.api as sm

#def mice()

class ImputedData:

    def __init__(self, data):
        self.data = pd.DataFrame(data)
        self.values = {}
        #self.cols = list(self.data.columns.values)
        for c in self.data.columns:
            temp = np.flatnonzero(pd.isnull(self.data[c]))
            l = len(temp)
            self.values[c] = np.zeros((2,l))
            self.values[c][0] = temp
                        
            self.values[c][1] = self.data[c].mean()

#            self.values[str(c)] = []
#            self.values[str(c)].append(self.data[str(c)][pd.isnull(self.data[str(c)])].index.tolist())

    def toDataFrame(self,copy = False):
       if copy: 
           df = pd.DataFrame(self.data)
           for k in self.values.keys():
               ix = self.values[k][0]
               v = self.values[k][1]
               df[k][ix] = v
           return df
       else:
           for k in self.values.keys():
               ix = self.values[k][0]
               v = self.values[k][1]
               self.data[k][ix] = v
    
#    def update(self):
#       for k in self.values.keys():
#           ix = self.values[k][0]
#           v = self.values[k][1]
#           self.data[k][ix] = v
    
    def toArray(self,copy = False):
        return np.asarray(self.toDataFrame(copy))
        

    def meanFillDataFrame(self):
        self.data = self.data.fillna(self.data.mean())
                
        # Class defining imputation for one variable.
    def updateValue(self,c,value):
        self.values[c][1] = np.asarray(value)
        
        
class Imputer:

    def __init__(self, data, formula, model_class,init_args={}, fit_args={}):

        # An imputed data instance, we need to 
        # change the init of ImputedData so it takes
        # data frame and creates a imputed data
        # based on the nan's.
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
#    def meanFillDataFrame(self):
#        self.data.fillna(self.data.mean())
#    def model_class(self):
#        def from_formula(formula,data,init_args = None):
    # Impute the dependent variable once
    def impute_asymptotic_bayes(self):
        #data1 = self.data.toDataFrame(True)
#        mc = self.model_class
        md = linear_model.OLS.from_formula(self.formula,self.data.data, **self.init_args)
        mdf = md.fit(**self.fit_args)
        params = mdf.params.copy()
        covmat = mdf.cov_params()
        covmat_sqrt = np.linalg.cholesky(covmat)
        u = np.random.chisquare(mdf.df_resid)
        sigstar = mdf.mse_resid*mdf.df_resid/u
        p = len(params)
        params += np.dot(covmat_sqrt, np.random.normal(0,sigstar,p))
#        mdf = copy.deepcopy(mdf)
        mdf.params = params
        ix = self.data.values[self.endog_name][0]
        ie = self.data.data.columns.get_loc(self.endog_name)
        ii = []
        for i in range(0,self.numexog):
            ii.append(self.data.data.columns.get_loc(self.exog_name[i])+1)
            if ie < ii[i]:
                ii[i]=ii[i] -1
        
            
        
        ix = ix.astype(int)
        ii = map(int,ii)
        #ii.astype(int)
#        icept = md.exog[ix,0]
#        exog = np.vstack((md.exog[ix,0],
        exog = md.exog[ix,ii]
#        exog = np.empty((len(ix),len(ii)))
#        for j in range(0,len(ii)):
#            for k in range(0,len(ix)):
#                exog[ix,ii] = md.exog[ix,ii]
        #exog.columns = ['x2']
        #what's this? .rvs()
        exog = pd.DataFrame(exog)
        exog.columns = [self.exog_name]
        new_endog = mdf.get_distribution(exog=exog,scale = sigstar)  
        self.data.updateValue(self.endog_name,new_endog)
        self.data.toDataFrame()       

# Manage a collection of imputers for variables in a  common dataframe.  
#This class does imputation and stores the imputed data sets, it does not fit 
#the analysis model.
class ImputerChain:

    def __init__(self, imputer_list):
        self.imputer_list = imputer_list
        self.imputer_list[0].data.meanFillDataFrame()
        

    # Impute each variable once, initialize missing values to column means
    def cycle(self):
        for im in self.imputer_list:
            im.impute_asymptotic_bayes() 


    # Impute data sets and save them to disk
    def generate_data(self, num, skip, base_name):
        for k in range(num):
            for j in range(skip):
                self.cycle()
            fname = "%s_%d.csv" % (base_name, k)
            self.data.toDataFrame().toCSV(fname)

#class ImputerChain:
#
#    def __init__(self, imputer_list):
#        self.imputer_list = imputer_list
#
#        self.data = self.imputer_list[0].data
#
#    # Impute each variable once
#    def cycle(self):
#        t = True
#        for im in self.imputer_list:
#            if t:
#                im.data.meanFillDataFrame()
#                temp = im.impute(self.data)
#            else:
#                im.impute(self.data)
#                t = False
#            self.data = im.data
#
#
#    # Impute data sets and save them to disk
#    def generate_data(self, num, skip, base_name):
#        for k in range(num):
#            for j in range(skip):
#                self.cycle()
#            fname = "%s_%d.csv" % (base_name, k)
#            self.data.toDataFrame().toCSV(fname)

    
# Manage a collection of imputers for a data set.  This class allows the imputers 
#to be run and the analysis model to be fit.  The resulting parameter estimates are 
#combined using the combining rule to produce the final results for the analysis.
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