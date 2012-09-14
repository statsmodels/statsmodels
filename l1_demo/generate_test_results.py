import statsmodels.api as sm
import scipy as sp
import pdb  # pdb.set_trace()

data = sm.datasets.spector.load()
data.exog = sm.add_constant(data.exog, prepend=True)
alpha = 3 * sp.array([0, 1, 1, 1])
res1 = sm.Logit(data.endog, data.exog).fit(disp=0, tol=1e-15)
res2 = sm.Logit(data.endog, data.exog).fit(method="l1", alpha=alpha, disp=0, acc=1e-10)

print res1.params
print res2.params
#print "params = \n" + str(res1.params)
#print "conf_int = \n" + str(res1.conf_int())
