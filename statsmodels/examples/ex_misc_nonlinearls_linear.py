import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import *

class Myfunc(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog

        x0, x1 = x[:,0], x[:,1]
        a, b, c = params
        return a + b*x0 + c*x1

    def jacobian(self,params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog

        x0, x1 = x[:,0], x[:,1]
        a, b, c = params
        a1 = np.ones(len(x0),)
        b1 = x0
        c1 = x1
        jacob = np.column_stack((a1,b1,c1))
        return jacob

x = np.arange(5.).repeat(2)  #[:,None] BUG
y = np.array([1, -2, 1, -2, 1.]).repeat(2)
sigma = np.array([1,  2, 1,  2, 1.]).repeat(2)

x = np.column_stack((x,0.1*x**2))
mod = Myfunc(y, x, sigma=sigma**2)
res = mod.fit(start_value=(0.042, 0.42,0.2))
res.df_model = 2. #subtract constant to agree with WLS
#print res.params
#print res.bse

resw = sm.WLS(y, sm.add_constant(x, prepend=True), weights=1./sigma**2).fit()
#print resw.params
#print resw.bse

#res3 = mod.fit(start_value=resw.params)
#print res3.params
#print res3.bse

#print res3.model.wexog
#print resw.model.wexog

print '\n\n\n'
print res.summary(yname='y', xname=['const', 'x0', 'x1'])
print res.view_iter() 
#print '\n\n\n'
#print resw.summary(yname='y', xname=['const', 'x0', 'x1'])

txt = res.summary(yname='y', xname=['const', 'x0', 'x1'])
txtw = resw.summary(yname='y', xname=['const', 'x0', 'x1'])
txt = str(txt)
txtw = str(txtw)
print txt[txt.find('No. Observations'):] == txtw[txtw.find('No. Observations'):]
