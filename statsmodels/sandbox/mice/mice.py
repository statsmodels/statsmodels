class ImputedData:

    def __init__(self, data):
        self.data = pd.DataFrame(self.data)
        #self.values = values
        self.missind = {}
        self.cols = list(self.data.columns.values)
        for c in self.cols:
            self.missind[str(c)] = self.data[str(c)][pd.isnull(self.data[str(c)])].index.tolist()        
        
#   def toDataFrame(self):
#       df = pd.DataFrame(self.data)
#       cols = list(df.columns.values)
#       for c in cols:
#           self.missind[str(c)] = df[str(c)][pd.isnull(df[str(c)])].index.tolist()
##       for k in self.values.keys():
##           ix = self.values[k][0]
##           v = self.values[k][1]
##           df[k][ix] = v
#       return df

#####worry about arrays later!!!!!!!!!!!!!!!!!!#######

#    def toArray(self):
#        ar = np.asarray(self.data.copy())
#        for k in self.values.keys():
#            ix = self.values[k][0]
#            v = self.values[k][1]
#            ar[ii,k] = v
#        return ar
        
    def meanFillDataFrame(self):
        df = pd.DataFrame(self.data)
        df.fillna(df.mean())
        return df        
        # Class defining imputation for one variable.
    
    
class Imputer:

    def __init__(self, data, formula, model_class,
                      init_args, fit_args):

        # An imputed data instance, we need to 
        # change the init of ImputedData so it takes
        # data frame and creates a imputed data
        # based on the nan's.
        self.data = ImputedData(data) 

        self.formula = formula
        self.model_class = model_class
        self.init_args = init_args
        self.fit_args = fit_args
        
        self.endog_name = str(self.formula.split("~")[0].strip())
        
    def meanFillDataFrame(self):
        self.data.fillna(self.data.mean())
        
    # Impute the dependent variable once
    def impute(self,data):
#        data1 = self.data.toDataFrame()
        md = self.model_class.from_formula(self.formula,data, args=**self.init_args)
        mdf = md.fit(**self.fit_args)
        ii = self.data.missind[self.exog_name]
        #ii = self.data.indices[self.endog_name]
        exog = md.exog[ii,:]
        new_endog = mdf.get_distribution(exog=exog)
        #what's this? .rvs()
        data[self.endog_name][ii] = new_endog

# Manage a collection of imputers for variables in a  common dataframe.  
#This class does imputation and stores the imputed data sets, it does not fit 
#the analysis model.
class ImputerChain:

    def __init__(self, imputer_list):
        self.imputer_list = imputer_list

    # Impute each variable once, initialize missing values to column means
    def cycle(self):
        t = True
        for im in self.imputer_list:
            if t:
                im.meanFillDataFrame()                
                im.impute(im.data) # for first variable, im.data is the mean-filled dataset
                t = False
            else:
                im.impute(self.data) #for subsequent variables, self.data is the dataset with the previous variable imputed
            self.data = im.data


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