import urllib.request
from io import StringIO

import statsmodels.api as sm
import numpy as np
import pandas as pd

# Read in data
url = "http://www.ats.ucla.edu/stat/data/hdp.csv"
httpstream = urllib.request.urlopen(url)
html = httpstream.read().decode('utf-8')
f = StringIO(html)
hdp = pd.read_csv(f)

# Generate GLMM model
fam = sm.families.Binomial()
glm_model = sm.MixedGLM.from_formula("remission ~ IL6", hdp, groups=hdp["DID"],family=fam)

# Fit GLMM model
x0=[0,0,1]
fit = glm_model.fit(start_params = x0)

