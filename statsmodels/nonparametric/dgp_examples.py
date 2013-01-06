# -*- coding: utf-8 -*-
"""Examples of non-linear functions for non-parametric regression

Created on Sat Jan 05 20:21:22 2013

Author: Josef Perktold
"""

import numpy as np




def fg1(x):
    '''Fan and Gijbels example function 1

    '''
    return x + 2 * np.exp(-16 * x**2)

def fg1eu(x):
    '''Eubank similar to Fan and Gijbels example function 1

    '''
    return x + 0.5 * np.exp(-50 * (x - 0.5)**2)

def fg2(x):
    '''Fan and Gijbels example function 2

    '''
    return np.sin(2 * x) + 2 * np.exp(-16 * x**2)

class UnivariateFanGijbels1_(object):
    '''Fan and Gijbels example function 1
    '''

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):

        if x is None:
            if distr_x is None:
                x = np.random.normal(loc=0, scale=1, size=nobs)
            else:
                x = distr_x.rvs(size=nobs)
        self.x = x
        self.x.sort()
        if distr_noise is None:
            noise = np.random.normal(loc=0, scale=0.07, size=nobs)
        else:
            noise = distr_noise.rvs(size=nobs)

        self.y_true = y_true = fg1(x)
        self.y = y_true + noise
        self.func = fg1

    def plot(self, scatter=True):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if scatter:
            ax.plot(self.x, self.y, 'o', alpha=0.6)

        xx = np.linspace(self.x.min(), self.x.max(), 200)
        ax.plot(xx, self.func(xx), lw=2, color='r')
        return fig

class _UnivariateFanGijbels(object):
    '''Fan and Gijbels example function 1
    '''

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):

        if x is None:
            if distr_x is None:
                x = np.random.normal(loc=0, scale=self.s_x, size=nobs)
            else:
                x = distr_x.rvs(size=nobs)
        self.x = x
        self.x.sort()
        if distr_noise is None:
            noise = np.random.normal(loc=0, scale=self.s_noise, size=nobs)
        else:
            noise = distr_noise.rvs(size=nobs)

        #self.func = fg1
        self.y_true = y_true = self.func(x)
        self.y = y_true + noise


    def plot(self, scatter=True):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if scatter:
            ax.plot(self.x, self.y, 'o', alpha=0.6)

        xx = np.linspace(self.x.min(), self.x.max(), 100)
        ax.plot(xx, self.func(xx), lw=2, color='r')
        return fig

class UnivariateFanGijbels1(_UnivariateFanGijbels):

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        self.s_x = 1.
        self.s_noise = 0.7
        self.func = fg1
        super(self.__class__, self).__init__(nobs=nobs, x=x,
                                                   distr_x=distr_x,
                                                   distr_noise=distr_noise)

class UnivariateFanGijbels2(_UnivariateFanGijbels):

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        self.s_x = 1.
        self.s_noise = 0.5
        self.func = fg2
        super(self.__class__, self).__init__(nobs=nobs, x=x,
                                                   distr_x=distr_x,
                                                   distr_noise=distr_noise)

class UnivariateFanGijbels1EU(_UnivariateFanGijbels):
    '''

    Eubank p.179f
    '''

    def __init__(self, nobs=50, x=None, distr_x=None, distr_noise=None):
        from scipy import stats
        distr_x = stats.uniform
        self.s_noise = 0.15
        self.func = fg1eu
        super(self.__class__, self).__init__(nobs=nobs, x=x,
                                                   distr_x=distr_x,
                                                   distr_noise=distr_noise)



if __name__ == '__main__':
    f = UnivariateFanGijbels1()
    fig = f.plot()
    fig.show()
    f = UnivariateFanGijbels2()
    fig = f.plot()
    fig.show()
    f = UnivariateFanGijbels1EU()
    fig = f.plot()
    #fig.show()

    from statsmodels.nonparametric.api import KernelReg

    model1 = KernelReg(endog=[f.y], exog=[f.x], reg_type='ll',
                      var_type='c', bw='cv_ls')
    mean1, mfx1 = model1.fit()
    fig.axes[0].plot(f.x, mean1)
    fig.show()


    f = UnivariateFanGijbels1()
    fig = f.plot()
    model = KernelReg(endog=[f.y], exog=[f.x], reg_type='ll',
                      var_type='c', bw='cv_ls')
    mean, mfx = model.fit()
    fig.axes[0].plot(f.x, mean)
    fig.show()
