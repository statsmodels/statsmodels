
if __name__ == '__main__': #pragma : no cover
    import statsmodels.api as sm
    from scipy.optimize.optimize import approx_fhess_p
    import numpy as np

    data = sm.datasets.spector.load()
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod = sm.Probit(data.endog, data.exog)
    res = mod.fit(method="newton")
    test_params = [1,0.25,1.4,-7]
    llf = mod.loglike
    score = mod.score
    hess = mod.hessian

    # below is Josef's scratch work

    def approx_hess_cs_old(x, func, args=(), h=1.0e-20, epsilon=1e-6):
        def grad(x):
            return approx_fprime_cs(x, func, args=args, h=1.0e-20)

        #Hessian from gradient:
        return (approx_fprime(x, grad, epsilon)
                + approx_fprime(x, grad, -epsilon))/2.


    def fun(beta, x):
        return np.dot(x, beta).sum(0)

    def fun1(beta, y, x):
        #print(beta.shape, x.shape)
        xb = np.dot(x, beta)
        return (y-xb)**2 #(xb-xb.mean(0))**2

    def fun2(beta, y, x):
        #print(beta.shape, x.shape)
        return fun1(beta, y, x).sum(0)

    nobs = 200
    x = np.arange(nobs*3).reshape(nobs,-1)
    x = np.random.randn(nobs,3)

    xk = np.array([1,2,3])
    xk = np.array([1.,1.,1.])
    #xk = np.zeros(3)
    beta = xk
    y = np.dot(x, beta) + 0.1*np.random.randn(nobs)
    xk = np.dot(np.linalg.pinv(x),y)


    epsilon = 1e-6
    args = (y,x)
    from scipy import optimize
    xfmin = optimize.fmin(fun2, (0,0,0), args)
    print(approx_fprime((1,2,3),fun,epsilon,x))
    jac = approx_fprime(xk,fun1,epsilon,args)
    jacmin = approx_fprime(xk,fun1,-epsilon,args)
    #print(jac)
    print(jac.sum(0))
    print('\nnp.dot(jac.T, jac)')
    print(np.dot(jac.T, jac))
    print('\n2*np.dot(x.T, x)')
    print(2*np.dot(x.T, x))
    jac2 = (jac+jacmin)/2.
    print(np.dot(jac2.T, jac2))

    hcs2 = approx_hess_cs(xk,fun2,args=args)
    print('hcs2')
    print(hcs2 - 2*np.dot(x.T, x))

    hfd3 = approx_hess(xk,fun2,args=args)
    print('hfd3')
    print(hfd3 - 2*np.dot(x.T, x))

    import numdifftools as nd
    hnd = nd.Hessian(lambda a: fun2(a, y, x))
    hessnd = hnd(xk)
    print('numdiff')
    print(hessnd - 2*np.dot(x.T, x))
    #assert_almost_equal(hessnd, he[0])
    gnd = nd.Gradient(lambda a: fun2(a, y, x))
    gradnd = gnd(xk)
