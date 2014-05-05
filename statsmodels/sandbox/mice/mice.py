class ImputedData:

    def __init__(self, data, values):
        self.data = data
        self.values = values
        #change to inser
   def toDataFrame(self):
       df = pd.DataFrame(self.data)
       for k in self.values.keys():
           ix = self.values[k][0]
           v = self.values[k][1]
           df[k][ix] = v
       return df

    def toArray(self):
        ar = np.asarray(self.data.copy())
        for k in self.values.keys():
            ix = self.values[k][0]
            v = self.values[k][1]
            ar[ii,k] = v
        return ar
        
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
        
        self.endog_name = self.formula.split("~")[0].strip()

    # Impute the dependent variable once
    def impute(self):
        data1 = self.data.toDataFrame()
        md = self.model_class.from_formula(self.formula,self.data1, args=**self.init_args)
        mdf = md.fit(**self.fit_args)
        ii = self.data.indices[self.endog_name]
        exog = md.exog[ii,:]
        new_endog = mdf.get_distribution(exog=exog).rvs()
        self.data.values[self.endog_name] = new_endog
 

# Manage a collection of imputers for variables in a  common dataframe.  
#This class does imputation and stores the imputed data sets, it does not fit 
#the analysis model.
class ImputerChain:

    def __init__(self, imputer_list):
        self.imputer_list = imputer_list

        self.data = self.imputer_list[0].data

    # Impute each variable once
    def cycle(self):
        for im in self.imputer_list:
            im.impute()

    # Impute data sets and save them to disk
    def generate_data(self, num, skip, base_name):

        for k in range(num):
            for j in range(skip):
                self.cycle()
            fname = "%s_%d.csv" % (base_name, k)
            self.data.toDataFrame().toCSV(fname)
    
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