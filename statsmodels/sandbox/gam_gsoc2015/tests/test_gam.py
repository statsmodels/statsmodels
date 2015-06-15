from smooth_basis import make_poly_basis
from gam import GamPenalty, LogitGam, GLMGam
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import minimize

sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))

def test_gam_penalty():
    """
    test the gam penalty class
    """

    n = 100000
    x = np.linspace(-10, 10, n)
    degree = 3
    basis, der_basis, der2_basis = make_poly_basis(x, degree)
    cov_der2 = np.dot(der2_basis.T, der2_basis)
    gp = GamPenalty(alpha=1, der2=der2_basis, cov_der2=cov_der2)
    params = np.array([1, 1, 1, 1])
    cost = gp.func(params) 
    # the integral between -10 and 10 of |2*a+6*b*x|^2 is 80*a^2 + 24000*b^2
    assert(int(cost*20) == 24080)

    params = np.array([1, 1, 0, 1])
    cost = gp.func(params) 
    assert(int(cost * 20) == 24000)

    params = np.array([1, 1, 2, 1])
    grad = gp.grad(params) * 20
    assert(int(grad[2]) == 320)
    assert(int(grad[3]) == 48000)

    return



def cost_function(params, alpha, basis, y):
    """
    :param params: an array of len 5
    :return: return the integral and the mean square error associated with the
    y = 2 * x * x * x - x polynomial where x goes from -1 to 1
    """
    a, b, c, d, e = params
    integral = 288*a*a/5 + 32 * a * c + 8 * (3 * b * b + c * c)
    integral = integral * 2 * alpha
    mse = np.linalg.norm(y - np.dot(basis, params))
    return mse, integral

def compute_cost(params, alpha, basis, y):
    mse, integral = cost_function(params, alpha, basis, y)
    return mse + integral


def sample_data():
    """

    :return:
        x, y a sample dataset to test univariate GAM,
        n the number of samples
        params0 the polynomial coefficients used to generate the y
    """
    n = 100
    x = np.linspace(-1, 1, n)
    y = 2 * x * x * x - x
    # the GAM penalty value is alpha * 48 * beta2 / 2
    params0 = np.array([0, -1, 0, 2, 0])
    return x, y, n, params0


def test_cost_function_and_gam_penalty():

    x, y, n, params0 = sample_data()

    degree = 4
    basis, der_basis, der2_basis = make_poly_basis(x, degree)
    cov_der2 = np.dot(der2_basis.T, der2_basis)

    for i in range(10):
        alpha = np.random.uniform(0, 1, 1)
        p1 = np.random.uniform([-1], [2], (1, 5))[0]
        gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
        mse, integral = cost_function(p1, alpha, basis, y)
        assert (int(gp.func(p1)) == int(integral), 'the func and the integral return different results')


def test_glm():

    x, y, n, params0 = sample_data()

    degree = 4
    basis, der_basis, der2_basis = make_poly_basis(x, degree)
    cov_der2 = np.dot(der2_basis.T, der2_basis)

    # test with 0 penalty
    alpha = 0
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
    glm_gam = GLMGam(y, basis, penal = gp)
    res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0,
                              disp=1, maxiter=10000, maxfun=5000, tol=1e-14)
    assert (np.linalg.norm(params0 - res_glm_gam.params) < 1e-13)
    opt_result0 = minimize(compute_cost, np.array([1, 1, 1, 1, 1]), args=(alpha, basis, y),
                           method='Powell')
    assert (np.linalg.norm(opt_result0['x'] - res_glm_gam.params) < 1.e-10)

    # equal value with penalty 1
    alpha = .1
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
    opt_result = minimize(compute_cost, np.array([1, 1, 1, 1, 1]), args=(alpha, basis, y))
    glm_gam = GLMGam(y, basis, penal=gp)
    res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0,
                              disp=1, maxiter=10000, maxfun=5000, tol=1e-14)

    print(compute_cost(res_glm_gam.params, alpha, basis, y), compute_cost(opt_result['x'], alpha, basis, y))
    print('gam score=', glm_gam.loglike(res_glm_gam.params), 'opt score=', glm_gam.loglike(opt_result['x']))

    approx_error = np.linalg.norm(np.dot(basis, res_glm_gam.params) - y)
    plt.plot(x, np.dot(basis, res_glm_gam.params), label='gam res')
    plt.plot(x, y, '.')
    plt.plot(x, np.dot(basis, opt_result['x']), label='opt res')
    plt.title('GLM alpha=' + str(alpha))
    plt.legend()
    plt.show()

    return


test_glm()
