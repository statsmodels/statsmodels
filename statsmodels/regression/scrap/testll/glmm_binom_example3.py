import urllib.request
from io import StringIO

import statsmodels.api as sm
from statsmodels.tools.numdiff import approx_fprime
import numpy as np
import pandas as pd

np.random.seed(seed=123)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

# Get data
url = "http://www.ats.ucla.edu/stat/data/hdp.csv"
httpstream = urllib.request.urlopen(url)
html = httpstream.read().decode('utf-8')
f = StringIO(html)
hdp = pd.read_csv(f)

# Generate Model
fam = sm.families.Binomial()
glm_model = sm.MixedGLM.from_formula("remission ~ IL6", hdp, groups=hdp["DID"],family=fam)


def test_grad():

    # Set some parameters
    params = np.random.normal(size=2)
    cov_re = np.abs(np.random.normal(size=(1,1)))

    # Get fn and its grad
    fungrad = glm_model._gen_joint_like_score(params, cov_re, glm_model._scale)
    def fun(x): return fungrad(x)[0]
    def grad(x): return fungrad(x)[1]

    # Evaluate fn and grad at arbitrary point
    ref = np.random.normal(size=(glm_model.n_groups, glm_model.k_re)).ravel()
    fp1 = approx_fprime(ref, fun)
    fp2 = grad(ref)

    try:
        np.testing.assert_allclose(fp1, fp2, rtol=1E-1,atol=0)
        print("All clear, no error.")
    except AssertionError as e:
        print("Numerical Derivative:\n"+str(fp1))
        print("Analytic Derivative:\n"+str(fp2))
        print("ref:\n"+str(ref))
        print("fe_params:\n"+str(params))
        print("cov_re:\n"+str(cov_re))
        adiff = np.abs(fp1-fp2)
        print("Rounded abs diff btwn grads:\n"+str(np.round(adiff,decimals=5)))
        print("Max absolute difference:\n" + str(max(adiff)))
        print(e)

for i in range(10):
    print("TESTING " + str(i))
    test_grad()