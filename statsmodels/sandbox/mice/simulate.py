from mice import ImputedData

def model_simulate(results, exog = None):
    self.results = results
    self.exog = exog
    if isinstance(results, glm) and isinstance(results.family, glm.family.binomial):
        _simulate_logistic_model(self.results,self.exog)
        #.... simulate binomial data
    elif isinstance(results, glm) and isinstance(results.family, glm.family.poisson):
        #.... simulate poisson data
    elif isinstance(results, discrete) and isinstance(results.family, discrete):
        #.... simulate some type of discrete data
    elif isinstance(results, ols):
        
        #.... simulate data from a linear model





def __simulate_logistic_model(results, exog = None):
    
    pass
def _simulate_linear_model(results, exog = None):
    pass
    
def _simulate_poisson_model(results, exog = None):
    pass
    
def _simulate_discrete_model(results, exog = None):
    pass