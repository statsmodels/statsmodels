# -*- coding: utf-8 -*-
"""

Created on Sun Jan 06 09:50:54 2013

Author: Josef Perktold
"""

if __name__ == '__main__':

    from statsmodels.nonparametric.api import KernelReg
    import statsmodels.nonparametric.dgp_examples as dgp
    import matplotlib.pyplot as plt

    from scipy import stats

    funcs = [dgp.UnivariateFanGijbels1(),
             dgp.UnivariateFanGijbels2(),
             dgp.UnivariateFanGijbels1EU(),
             #dgp.UnivariateFanGijbels2(distr_x=stats.uniform(-2, 4))
             dgp.UnivariateFunc1()
             ]

    res = []
    fig = plt.figure()
    for i,func in enumerate(funcs):
        #f = func()
        f = func
        model = KernelReg(endog=[f.y], exog=[f.x], reg_type='ll',
                          var_type='c', bw='cv_ls')
        mean, mfx = model.fit()
        ax = fig.add_subplot(2, 2, i+1)
        f.plot(ax=ax)
        ax.plot(f.x, mean, color='r', lw=2, label='est. mean')
        ax.legend(loc='upper left')
        res.append((model, mean, mfx))

    fig.suptitle('Kernel Regression')
    fig.show()
