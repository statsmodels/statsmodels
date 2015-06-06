
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from smooth_basis import make_poly_basis
from gam import GamPenalty
import numpy as np

def test_gam_penalty():
    ''' test the gam penalty class '''
    n = 100000
    x = np.linspace(-10, 10, n)
    degree = 3
    basis, der_basis, der2_basis = make_poly_basis(x, degree)
    cov_der2 = np.dot(der2_basis.T, der2_basis)
    gp = GamPenalty(alpha=1, der2=der2_basis, cov_der2=cov_der2)
    params = np.array([1, 1, 1, 1])
    cost = gp.func(params) 
    # the integral between -10 and 10 of |2*a+6*b*x|^2 is 80*a^2 + 24000*b^2
    assert(int(cost/n*20) == 24080)

    params = np.array([1, 1, 0, 1])
    cost = gp.func(params) 
    assert(int(cost/n*20) == 24000)

    params = np.array([1, 1, 2, 1])
    grad = gp.grad(params)/n*20
    assert(int(grad[2]) == 320)
    assert(int(grad[3]) == 48000)

    return
