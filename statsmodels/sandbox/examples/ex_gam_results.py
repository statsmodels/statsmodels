# -*- coding: utf-8 -*-
"""Example results for GAM from tests

Created on Mon Nov 07 13:13:15 2011

Author: Josef Perktold

The example is loaded from a test module. The test still fails but the
results look relatively good.
I don't know yet why there is the small difference and why GAM doesn't
converge in this case

"""



from statsmodels.sandbox.tests.test_gam import _estGAMGaussianLogLink


tt = _estGAMGaussianLogLink()
comp, const = tt.res_gam.smoothed_demeaned(tt.mod_gam.exog)
comp_glm_ = tt.res2.model.exog * tt.res2.params
comp1 = comp_glm_[:,1:4].sum(1)
mean1 = comp1.mean()
comp1 -= mean1
comp2 = comp_glm_[:,4:].sum(1)
mean2 = comp2.mean()
comp2 -= mean2

comp1_true = tt.res2.model.exog[:,1:4].sum(1)
mean1 = comp1_true.mean()
comp1_true -= mean1
comp2_true = tt.res2.model.exog[:,4:].sum(1)
mean2 = comp2_true.mean()
comp2_true -= mean2

noise = tt.res2.model.endog - tt.mu_true
noise_eta =  tt.family.link(tt.res2.model.endog) - tt.y_true

import matplotlib.pyplot as plt
plt.figure()
plt.plot(noise, 'k.')
plt.figure()
plt.plot(comp, 'r-')
plt.plot(comp1, 'b-')
plt.plot(comp2, 'b-')
plt.plot(comp1_true, 'k--', lw=2)
plt.plot(comp2_true, 'k--', lw=2)
#the next doesn't make sense - non-linear
#c1 = tt.family.link(tt.family.link.inverse(comp1_true) + noise)
#c2 = tt.family.link(tt.family.link.inverse(comp2_true) + noise)
#not nice in example/plot: noise variance is constant not proportional
plt.plot(comp1_true + noise_eta, 'g.', alpha=0.95)
plt.plot(comp2_true + noise_eta, 'r.', alpha=0.95)
#plt.plot(c1, 'g.', alpha=0.95)
#plt.plot(c2, 'r.', alpha=0.95)
plt.title('Gaussian loglink, GAM (red), GLM (blue), true (black)')
plt.show()

