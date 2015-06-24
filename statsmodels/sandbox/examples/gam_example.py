import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import make_poly_basis, make_bsplines_basis
from statsmodels.sandbox.gam_gsoc2015.gam import GamPenalty, LogitGam, GLMGam, MultivariateGamPenalty
import statsmodels.api as sm

sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))


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
basis, der_basis, der2_basis = make_bsplines_basis(x, df, degree)
cov_der2 = np.dot(der2_basis.T, der2_basis)


# train the gam logit model ##
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

# %%%%%%%%%%% Logit 2 #####################


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


# GAM GLM

# y is continuous
n = 200
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


# %%%%%%%%%%%%%%%%%% Multivariate GAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

basis1, der_basis1, der2_basis1 = make_poly_basis(x1, degree1, intercept=False)
basis2, der_basis2, der2_basis2 = make_poly_basis(x2, degree2, intercept=False)

basis = np.hstack([basis1, basis2])
der_basis = [der_basis1, der_basis2]
der2_basis = [der2_basis1, der2_basis2]
cov_der2 = [np.dot(der2_basis1.T, der2_basis1),
            np.dot(der2_basis2.T, der2_basis2)]

alpha = [0, 0]
wts = [1, 1]
mgp = MultivariateGamPenalty(wts=wts, alphas=alpha, cov_der2=cov_der2,
                             der2=der2_basis)

mLG = LogitGam(y, basis, penal=mgp)
res_mLG = mLG.fit(maxiter=1000, tol=1e-13)

param1 = res_mLG.params[mgp.mask[0]]
param2 = res_mLG.params[mgp.mask[1]]
param = res_mLG.params


plt.subplot(3, 2, 1)
plt.title('alpha=' + str(alpha))
plt.plot(x1, np.dot(basis1, param1), label='x1')
plt.legend()
plt.subplot(3, 2, 3)
plt.plot(x2, np.dot(basis2, param2), label='x2')
plt.legend()
plt.subplot(3, 2, 5)
plt.plot(sigmoid(np.dot(basis, param)), label=r'$\hat y$')
plt.plot(y, '*', label='y')
plt.ylim(-1, 2)
plt.legend()

alpha = [.1, .2]
wts = [1, 1]

mgp = MultivariateGamPenalty(wts=wts, alphas=alpha, cov_der2=cov_der2,
                             der2=der2_basis)

mLG = LogitGam(y, basis, penal=mgp)
res_mLG = mLG.fit(maxiter=1000, tol=1e-13)

param1 = res_mLG.params[mgp.mask[0]]
param2 = res_mLG.params[mgp.mask[1]]
param = res_mLG.params


# plot with different alpha
plt.subplot(3, 2, 2)
plt.title('alpha=' + str(alpha))
plt.plot(x1, np.dot(basis1, param1), label='x1')
plt.legend()
plt.subplot(3, 2, 4)
plt.plot(x2, np.dot(basis2, param2), label='x2')
plt.legend()
plt.subplot(3, 2, 6)
plt.plot(sigmoid(np.dot(basis, param)), label=r'$\hat y$')
plt.plot(y, '*', label='y')
plt.ylim(-1, 2)
plt.legend()
plt.show()

