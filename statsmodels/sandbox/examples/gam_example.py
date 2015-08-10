__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '08/07/15'


import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (UnivariateBSplines, UnivariatePolynomialSmoother, BSplines,
                                                           PolynomialSmoother, CubicSplines)
from statsmodels.sandbox.gam_gsoc2015.gam import UnivariateGamPenalty, LogitGam, GLMGam, MultivariateGamPenalty
import statsmodels.api as sm
from statsmodels.sandbox.gam_gsoc2015.gam_cross_validation.gam_cross_validation import MultivariateGAMCV, MultivariateGAMCVPath
from statsmodels.sandbox.gam_gsoc2015.gam_cross_validation.cross_validators import KFold

sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))



###############################################

example = None  #'ex1'
if example == 'ex1':
    print(example)

    n = 100
    # make the data
    x = np.linspace(-10, 10, n)
    y = 1/(1 + np.exp(-x*x))
    mu = y.mean()
    y[y > mu] = 1
    y[y < mu] = 0

    # make the splines basis ##
    df = 10
    degree = 5
    x = x - x.mean()

    univ_bsplines = UnivariateBSplines(x, degree=degree, df=df)

    # train the gam logit model ##
    alphas = [0, 0.1, 1, 10]

    for i, alpha in enumerate(alphas):
        plt.subplot(2, 2, i+1)
        params0 = np.random.normal(0, 1, df)
        gp = UnivariateGamPenalty(univ_bsplines, wts=1, alpha=alpha)
        g = LogitGam(y, univ_bsplines.basis_, penal=gp)
        res_g = g.fit()
        plt.plot(x, sigmoid(np.dot(univ_bsplines.basis_, res_g.params)))
        plt.plot(x, y, '.')
        plt.ylim(-1, 2)
        plt.title('alpha=' + str(alpha))
    plt.show()


if example == 'ex2':
    # Logit 2
    print(example)

    spector_data = sm.datasets.spector.load()
    spector_data.exog = sm.add_constant(spector_data.exog)

    y = spector_data.endog
    X = spector_data.exog
    x = X[:, 2]

    x = x - x.mean()
    degree = 4
    univ_bsplines = UnivariateBSplines(x, degree=degree, df=df)
    for i, alpha in enumerate(alphas):
        gp = UnivariateGamPenalty(univ_bsplines, alpha=alpha)
        gam = LogitGam(y, univ_bsplines.basis_, penal = gp)
        res_gam = gam.fit(method='nm', max_start_irls=0,
                          disp=1, maxiter=5000, maxfun=5000)
        plt.subplot(2, 2, i+1)
        plt.plot(x, sigmoid(np.dot(univ_bsplines.basis_, res_gam.params)), 'o')
        plt.plot(x, y, '.')
        plt.title('alpha=' + str(alpha))
        plt.ylim(-1, 2)
    plt.show()


if example == 'ex3':
    # GAM GLM
    print(example)

    # y is continuous
    n = 200
    x = np.linspace(-10, 10, n)
    y = x * x + np.random.normal(0, 5, n)
    y -= y.mean()

    x = x - x.mean()
    univ_bsplines = UnivariateBSplines(x, degree=degree, df=df)
    plt.figure()
    alphas = [0, 0.001, 0.01, 100]
    for i, alpha in enumerate(alphas):
        plt.subplot(2, 2, i+1)

        # train the model
        gp = UnivariateGamPenalty(univ_bsplines, alpha=alpha)
        glm_gam = GLMGam(y, univ_bsplines.basis_, penal = gp)
        res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0,
                                  disp=1, maxiter=5000, maxfun=5000)
        plt.plot(x, np.dot(univ_bsplines.basis_, res_glm_gam.params))
        plt.plot(x, y, '.')
        plt.title('GLM alpha=' + str(alpha))
    plt.show()

if example == 'ex4':
    # Multivariate GAM
    print(example)

    n = 100
    x1 = np.sort(np.random.uniform(-5, 5, n))
    x2 = np.sort(np.random.uniform(0, 10, n))
    poly = x1*x1 + x2 + np.random.normal(0, 0.01, n)
    y = sigmoid(poly)
    mu = y.mean()
    yc = y.copy()
    y[y > mu] = 1
    y[y <= mu] = 0

    degree1 = 3
    degree2 = 4
    x = np.vstack([x1, x2]).T
    bsplines = BSplines(x, [df, df], [degree1, degree2])
    alpha = [0, 0]

    mgp = MultivariateGamPenalty(bsplines, alphas=alpha, wts=[1, 1])

    mLG = LogitGam(y, bsplines.basis_, penal=mgp)
    res_mLG = mLG.fit(maxiter=1000, tol=1e-13)

    # TODO: this does not work.
    res_mLG.plot_partial()


    alpha = [.1, .2]
    wts = [1, 1]

    mgp = MultivariateGamPenalty(bsplines, wts=wts, alphas=alpha)

    mLG = LogitGam(y, bsplines.basis_, penal=mgp)
    res_mLG = mLG.fit(maxiter=1000, tol=1e-13)

    res_mLG.plot_partial()  # TODO: partial_plot is not working

if example == 'ex5':
    print(example)

    # Multivariate GLMGam.
    n = 200
    x = np.zeros(shape=(n, 2))
    x[:, 0] = np.linspace(-10, -5, n)
    x[:, 1] = np.linspace(5, 10, n)

    y = x[:, 0]**3 + x[:, 1]**2 + np.random.normal(0, 10, n)
    poly = PolynomialSmoother(x, degrees=[5, 5])

    gp = MultivariateGamPenalty(poly, alphas=[0, 0], wts=[1, 1])
    gam = GLMGam(y, poly.basis_, penal=gp)
    gam_ris = gam.fit()


    gam_ris.plot_partial(poly, plot_se=False)
    plt.show()
    print(gam_ris.params)

    plt.plot(y, '.')
    plt.plot(gam_ris.predict(poly.basis_, '.'))
    plt.show()

if example == 'ex6':
    print(example)

    # Multivariate Gam CV
    def cost(y1, y2):
        return np.linalg.norm(y1 - y2) / len(y2)

    cv = KFold(k_folds=5, shuffle=True)
    gam_cv = MultivariateGAMCV(poly, alphas=[1, 1], gam=GLMGam, cost=cost, y=y, cv=cv)
    ris_gam_cv = gam_cv.fit()

    print('Cross validation error=', ris_gam_cv)

    # Multivariate GAM CV path

    alphas = np.linspace(0, 10, 5)
    gam_cv_path = MultivariateGAMCVPath(poly, alphas=[alphas, alphas], gam=GLMGam, cost=cost, y=y, cv=cv).fit()

    print(gam_cv_path.alpha_cv_, gam_cv_path.cv_error_)

example = 'ex7'
if example == 'ex7':
    print(example)

    # GAM PIRLS
    from statsmodels.sandbox.gam_gsoc2015.smooth_basis import CubicSplines

    n = 500
    x = np.random.uniform(-1, 1, n)

    y = 10*x**3 - 10*x + np.random.normal(0, 1, n)

    y -= y.mean()
    cs = CubicSplines(x, 10)

    # required only to initialize the gam. they have no influence on the result.
    dummy_smoother = PolynomialSmoother(x, [2])
    gp = MultivariateGamPenalty(dummy_smoother, alphas=[0])
    for i, alpha in enumerate([0, 1, 5, 10]):

        gam = GLMGam(y, cs.basis_, penal=gp)
        gam_res = gam._fit_pirls(y=y, spl_x=cs.basis_, spl_s=cs.s, alpha=alpha)
        y_est = gam_res.predict(cs.basis_)

        plt.subplot(2, 2, i+1)
        plt.plot(x, y, '.')
        plt.plot(x, y_est, '.')
        plt.title('alpha=' + str(alpha))

    plt.show()

example = 'ex8'
if example == 'ex8':
    print(example)
    # Multivariate GAM PIRLS
    n = 500
    x1 = np.random.uniform(-1, 1, n)
    y1 = 10*x1**3 - 10*x1 + np.random.normal(0, 1, n)
    y1 -= y1.mean()
    x2 = np.random.uniform(-1, 1, n)
    y2 = x2**2 + np.random.normal(0, 0.1, n)
    y2 -= y2.mean()

    Y = y1 + y2
    cs1 = CubicSplines(x1, 10)
    cs2 = CubicSplines(x2, 10)

    spl_X = np.hstack([cs1.basis_, cs2.basis_])
    spl_S = [cs1.s, cs2.s]

    n_var1 = cs1.basis_.shape[1]
    n_var2 = cs2.basis_.shape[1]

    dummy_smoother = BSplines(x1, dfs=[10], degrees=[3])
    gp = MultivariateGamPenalty(dummy_smoother, alphas=[1])
    gam = GLMGam(y1, cs1.basis_, penal=gp)

    i = 0
    for alpha in [0, 1, 2]:

        gam_results = gam._fit_pirls(Y, spl_X, spl_S, alpha=[alpha]*2)
        y1_est = np.dot(cs1.basis_, gam_results.params[:n_var1])
        y2_est = np.dot(cs2.basis_, gam_results.params[n_var1:])

        y1_est -= y1_est.mean() # TODO: the estimate is good but has a very large mean. Why is this happening?
        y2_est -= y2_est.mean()

        i += 1
        plt.subplot(3, 2, i)
        plt.title('x1  alpha=' + str(alpha))
        plt.plot(x1, y1, '.', label='Real', c='green')
        plt.plot(x1, y1_est, '.', label='Estimated', c='blue')

        i += 1
        plt.subplot(3, 2, i)
        plt.title('x2  alpha=' + str(alpha))
        plt.plot(x2, y2, '.', label='Real', c='green')
        plt.plot(x2, y2_est, '.', label='Estimated', c='blue')

    plt.tight_layout()
    plt.show()

#example = 'ex9'
if example == 'ex9':
    print(example)
    # PIRLS ###
    n = 500
    x = np.random.uniform(-1, 1, n)

    y = 10*x**3 - 10*x + np.random.normal(0, 1, n)

    y -= y.mean()

    # required only to initialize the gam. they have no influence on the result.
    smoother = UnivariatePolynomialSmoother(x,  degree=4)
    smoother = UnivariateBSplines(x, degree=4, df=10)

    gp = UnivariateGamPenalty(smoother, alpha=0)
    gam = GLMGam(y, smoother.basis_, penal=gp)

    for i, alpha in enumerate([0, .001, .01, .1]):
        gam_res = gam._fit_pirls(y=y, spl_x=smoother.basis_, spl_s=smoother.cov_der2_, alpha=alpha)

        y_est = np.dot(smoother.basis_, gam_res.params.T)

        plt.subplot(2, 2, i+1)
        plt.plot(x, y, '.')
        plt.plot(x, y_est, '.')
        plt.title('alpha=' + str(alpha))

    plt.tight_layout()
    plt.show()
