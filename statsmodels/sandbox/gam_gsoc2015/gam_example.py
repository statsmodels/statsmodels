import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import PenalizedMixin, GamPenalty, LogitGam, GLMGam
import statsmodels.api as sm

sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))


n = 100

# make the data
x = np.linspace(-10, 10, n)
y = 1/(1 + np.exp(-x*x))
mu = y.mean()
y[y > mu] = 1
y[y < mu] = 0

## make the splines basis ##
df = 10
degree = 5
x = x - x.mean()
basis, der_basis, der2_basis = make_bsplines_basis(x, df, degree)
cov_der2 = np.dot(der2_basis.T, der2_basis)


## train the gam logit model ##
alphas = [0, 0.1, 1, 10]

for i, alpha in enumerate(alphas):
    plt.subplot(2, 2, i+1)
    params0 = np.random.normal(0, 1, df)
    gp = GamPenalty(wts=1, alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
    g = LogitGam(y, basis, penal=gp)
    res_g = g.fit()
    plt.plot(x, sigmoid(np.dot(basis, res_g.params)))
    plt.plot(x, y, '.')
    plt.ylim(-1, 2)
    plt.title('alpha=' + str(alpha))
plt.show()

####### Logit 2 #####################


spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog)

y = spector_data.endog
X = spector_data.exog
x = X[:, 2]

x = x - x.mean()
degree = 4
basis, der_basis, der2_basis = make_bsplines_basis(x, df, degree)
cov_der2 = np.dot(der2_basis.T, der2_basis)
for i, alpha in enumerate(alphas):
    gp = GamPenalty(alpha=alpha, der2=der2_basis, cov_der2=cov_der2)
    gam = LogitGam(y, basis, penal = gp)
    res_gam = gam.fit(method='nm', max_start_irls=0,
                      disp=1, maxiter=5000, maxfun=5000)
    plt.subplot(2, 2, i+1)
    plt.plot(x, sigmoid(np.dot(basis, res_gam.params)), 'o')
    plt.plot(x, y, '.')
    plt.title('alpha=' + str(alpha))
    plt.ylim(-1, 2)
plt.show()


######## GAM GLM  ##################

# y is continuous
n = 50
x = np.linspace(-10, 10, n)
y = x * x + np.random.normal(0, 1, n)
y -= y.mean()

x = x - x.mean()
basis, der_basis, der2_basis = make_bsplines_basis(x, df, degree)
cov_der2 = np.dot(der2_basis.T, der2_basis)

plt.figure()
alphas = [0, 0.001, 0.01, 100]
for i, alpha in enumerate(alphas):
    plt.subplot(2, 2, i+1)

    # train the model
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
    glm_gam = GLMGam(y, basis, penal = gp)
    res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0,
                              disp=1, maxiter=5000, maxfun=5000)

    plt.plot(x, np.dot(basis, res_glm_gam.params))
    plt.plot(x, y, '.')

    plt.title('GLM alpha=' + str(alpha))
plt.show()
# despite the large alpha we don't see a penalization



########### Josef's version of the GLM example ##############################
# y is continuous
y = x * x + np.random.normal(0, 1, n)
y -= y.mean()
basis_dm = basis - basis.mean(0)

plt.figure()
alphas = [0, 0.001, 0.01, 1]
for i, alpha in enumerate(alphas):
    plt.subplot(2, 2, i+1)

    # train the model
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
    glm_gam = GLMGam(y, basis_dm, penal = gp)
    #res_glm_gam = glm_gam.fit(method='bfgs', disp=1)
    res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0, disp=1, maxiter=6000, maxfun=5000)

    plt.plot(x, np.dot(basis, res_glm_gam.params))
    plt.plot(x, y, 'o')

    plt.title('GLM alpha=' + str(alpha))
plt.show()
# despite the large alpha we don't see a penalization
