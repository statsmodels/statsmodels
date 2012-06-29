import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

class funcMisra1a_J(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return b1*(1-np.exp(-b2*x))

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2 = params
        return np.column_stack([1-np.exp(-b2*x),b1*x*np.exp(-b2*x)])

x = np.array([77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4,
              434.8, 477.3, 536.8, 593.1, 689.1, 760.0])
y = np.array([10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82,
              50.76, 55.05, 61.01, 66.4, 75.47, 81.78])
mod1 = funcMisra1a_J(y, x,weights=(1,1,1,1,1,1,1,
                                   0.49,0.49,.49,.49,.49,.49,.49))
res_start1 = mod1.fit(start_value=[500.0, 0.0001])
print '\n\n'
print res_start1.summary()
print '\n\n'
print res_start1.view_iter(parameters=['b1','b2'])
print '\n\n'
print res_start1.prediction_table(alpha=0.08)
