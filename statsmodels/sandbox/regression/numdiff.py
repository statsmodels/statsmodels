'''numerical differentiation function, gradient, Jacobian, and Hessian

These are simple forward differentiation, so that we have them available
without dependencies.

* Jacobian should be faster than numdifftools because it doesn't use loop over observations.
* numerical precision will vary and depend on the choice of stepsizes

Todo:
* some cleanup
* check numerical accuracy (and bugs) with numdifftools and analytical derivatives
  - linear least squares case: (hess - 2*X'X) is 1e-8 or so
  - gradient and Hessian agree with numdifftools when evaluated away from minimum
  - forward gradient, Jacobian evaluated at minimum is inaccurate, centered (+/- epsilon) is ok
* dot product of Jacobian is different from Hessian, either wrong example or a bug (unlikely),
  or a real difference


What are the conditions that Jacobian dotproduct and Hessian are the same?
see also:
BHHH: Greene p481 17.4.6,  MLE Jacobian = d loglike / d beta , where loglike is vector for each observation
   see also example 17.4 when J'J is very different from Hessian
   also does it hold only at the minimum, what's relationship to covariance of Jacobian matrix
http://projects.scipy.org/scipy/ticket/1157
http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
   objective: sum((y-f(beta,x)**2),   Jacobian = d f/d beta   and not d objective/d beta as in MLE Greene
   similar: http://crsouza.blogspot.com/2009/11/neural-network-learning-by-levenberg_18.html#hessian

in example: if J = d x*beta / d beta then J'J == X'X
   similar to http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

Author : josef-pkt
License : BSD

'''



import numpy as np

EPS = 2.2204460492503131e-016

#from scipy.optimize
def approx_fprime(xk,f,epsilon=1e-8,*args):
    f0 = f(*((xk,)+args))
    grad = np.zeros((len(xk),), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = epsilon
        grad[k] = (f(*((xk+ei,)+args)) - f0)/epsilon
        ei[k] = 0.0
    return grad

def approx_fprime1(xk, f, epsilon=1e-12, args=(), centered=False):
    '''
    Gradient of function, or Jacobian if function f returns 1d array

    Parameters
    ----------
    xk : array
        parameters at which the derivative is evaluated
    f : function
        `*((xk,)+args)` returning either one value or 1d array
    epsilon : float
        stepsize, TODO add default
    *args : tuple
        tuple of additional arguments for function f

    Returns
    -------
    grad : array
        gradient or Jacobian, evaluated with single step forward differencing

    Notes
    -----

    todo:
    * add scaled stepsize
    ss- should scaled stepsize be an option in the gradient or should the
    gradient be scaled in within an optimization framework?
    '''
    f0 = f(*((xk,)+args))
    nobs = len(np.atleast_1d(f0)) # it could be a scalar
    grad = np.zeros((nobs,len(xk)), float)
    ei = np.zeros((len(xk),), float)
    #centered = False
    if not centered:
        for k in range(len(xk)):
            ei[k] = epsilon
            grad[:,k] = (f(*((xk+ei,)+args)) - f0)/epsilon
            ei[k] = 0.0
    else:
        for k in range(len(xk)):
            ei[k] = epsilon/2.
            grad[:,k] = (f(*((xk+ei,)+args)) - f(*((xk-ei,)+args)))/epsilon
            ei[k] = 0.0
    return grad.squeeze()

def approx_hess(xk, f, epsilon=None, args=()):#, returngrad=True):
    '''
    Calculate Hessian and Gradient by forward differentiation

    todo: cleanup args and options
    '''
    returngrad=True
    if epsilon is None:  #check
        #eps = 1e-5
        eps = 2.2204460492503131e-016 #np.MachAr().eps
        step = None
    else:
        step = epsilon  #TODO: shouldn't be here but I need to figure out args

    #x = xk  #alias drop this


    # Compute the stepsize (h)
    if step is None:  #check
        h = eps**(1/3.)*np.maximum(np.abs(xk),1e-2)
    else:
        h = step
    xh = xk + h
    h = xh - xk
    ee = np.diag(h.ravel())

    f0 = f(*((xk,)+args))
    # Compute forward step
    n = len(xk)
    g = np.zeros(n);
    for i in range(n):
        g[i] = f(*((xk+ee[i,:],)+args))

    hess = np.outer(h,h)
##    print hess.shape,
##    print 'H=', hess
##    print 'h=', hess
    # Compute "double" forward step
    for i in range(n):
        for j in range(i,n):
            hess[i,j] = (f(*((xk+ee[i,:]+ee[j,:],)+args))-g[i]-g[j]+f0)/hess[i,j]
            hess[j,i] = hess[i,j]
    if returngrad:
        grad = (g - f0)/h
        return hess, grad
    else:
        return hess

def approx_hess3(x0, f, epsilon=None, args=()):
    '''calculate Hessian with finite difference derivative approximation

    Parameters
    ----------
    x0 : array_like
       value at which function derivative is evaluated
    f : function
       function of one array f(x)
    epsilon : float
       stepsize, if None, then stepsize is automatically chosen

    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian

    Notes
    -----
    based on equation 9 in
    M. S. RIDOUT: Statistical Applications of the Complex-step Method
    of Numerical Differentiation, University of Kent, Canterbury, Kent, U.K.

    The stepsize is the same for the complex and the finite difference part.
    '''

    if epsilon is None:
        h = EPS**(1/5.)*np.maximum(np.abs(x0),1e-2) # 1/4 from ...
    else:
        h = epsilon
    xh = x0 + h
    h = xh - x0
    ee = np.diag(h)
    hess = np.outer(h,h)

    n = dim = np.size(x0) #TODO: What's the assumption on the shape here?

    for i in range(n):
        for j in range(i,n):
            hess[i,j] = (f(*((x0 + ee[i,:] + ee[j,:],)+args))
                            - f(*((x0 + ee[i,:] - ee[j,:],)+args))
                         - (f(*((x0 - ee[i,:] + ee[j,:],)+args))
                            - f(*((x0 - ee[i,:] - ee[j,:],)+args)))
                         )/4./hess[i,j]
            hess[j,i] = hess[i,j]

    return hess

def approx_fhess_p(x0,p,fprime,epsilon,*args):
    """
    Approximate the Hessian when the Jacobian is available.

    Parameters
    ----------
    x0 : array-like
        Point at which to evaluate the Hessian
    p : array-like
        Point
    fprime : func
        The Jacobian function
    epsilon : float

    """
    f2 = fprime(*((x0+epsilon*p,)+args))
    f1 = fprime(*((x0,)+args))
    return (f2 - f1)/epsilon

#copied from try_gradient_complexsteps, which contains notes and some examples for testing
#renamed from complex_step_grad to approx_fprime_cs, change order of arguments
#from Guilherme P. de Freitas, numpy mailing list
def approx_fprime_cs(x0, f, args=(), h=1.0e-20):
    '''calculate gradient or Jacobian with complex step derivative approximation

    Parameters
    ----------
    f : function
       function of one array f(x)
    x : array_like
       value at which function derivative is evaluated

    Returns
    -------
    partials : ndarray
       array of partial derivatives, Gradient or Jacobian
    '''
    dim = np.size(x0) #TODO: What's the assumption on the shape here?
    increments = np.identity(dim) * 1j * h
    #TODO: see if this can be vectorized, but usually dim is small
    partials = [f(x0+ih, *args).imag / h for ih in increments]
    return np.array(partials).T

def approx_hess_cs(x0, func, args=(), h=1.0e-20, epsilon=1e-6):
    def grad(x0):
        return approx_fprime_cs(x0, func, args=args, h=1.0e-20)

    #Hessian from gradient:
    return (approx_fprime1(x0, grad, epsilon)
            + approx_fprime1(x0, grad, -epsilon))/2.


def approx_hess_cs2(x0, f, epsilon=None, args=()):
    '''calculate Hessian with complex step (and fd) derivative approximation

    Parameters
    ----------
    x0 : array_like
       value at which function derivative is evaluated
    f : function
       function of one array f(x)
    epsilon : float
       stepsize, if None, then stepsize is automatically chosen

    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian

    Notes
    -----
    based on equation 10 in
    M. S. RIDOUT: Statistical Applications of the Complex-step Method
    of Numerical Differentiation, University of Kent, Canterbury, Kent, U.K.

    The stepsize is the same for the complex and the finite difference part.
    '''

    if epsilon is None:
        h = EPS**(1/5.)*np.maximum(np.abs(x0),1e-2) # 1/4 from ...
    else:
        h = epsilon
    xh = x0 + h
    h = xh - x0
    ee = np.diag(h)
    hess = np.outer(h,h)

    n = dim = np.size(x0) #TODO: What's the assumption on the shape here?

    for i in range(n):
        for j in range(i,n):
            hess[i,j] = (f(*((x0 + 1j*ee[i,:] + ee[j,:],)+args))
                         - f(*((x0 + 1j*ee[i,:] - ee[j,:],)+args))).imag/2./hess[i,j]
            hess[j,i] = hess[i,j]

    return hess


def fun(beta, x):
    return np.dot(x, beta).sum(0)

def fun1(beta, y, x):
    #print beta.shape, x.shape
    xb = np.dot(x, beta)
    return (y-xb)**2 #(xb-xb.mean(0))**2

def fun2(beta, y, x):
    #print beta.shape, x.shape
    return fun1(beta, y, x).sum(0)


if __name__ == '__main__':
    import statsmodels.api as sm
    from scipy.optimize.optimize import approx_fhess_p
    import numpy as np

    data = sm.datasets.spector.load()
    data.exog = sm.add_constant(data.exog)
    mod = sm.Probit(data.endog, data.exog)
    res = mod.fit(method="newton")
    test_params = [1,0.25,1.4,-7]
    llf = mod.loglike
    score = mod.score
    hess = mod.hessian

# below is Josef's scratch work

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
    print approx_fprime((1,2,3),fun,epsilon,x)
    jac = approx_fprime1(xk,fun1,epsilon,args)
    jacmin = approx_fprime1(xk,fun1,-epsilon,args)
    #print jac
    print jac.sum(0)
    print '\nnp.dot(jac.T, jac)'
    print np.dot(jac.T, jac)
    print '\n2*np.dot(x.T, x)'
    print 2*np.dot(x.T, x)
    jac2 = (jac+jacmin)/2.
    print np.dot(jac2.T, jac2)

    #he = approx_hess(xk,fun2,epsilon,*args)
    print approx_hess(xk,fun2,1e-3,args)
    he = approx_hess(xk,fun2,None,args)
    print 'hessfd'
    print he
    print 'epsilon =', None
    print he[0] - 2*np.dot(x.T, x)

    for eps in [1e-3,1e-4,1e-5,1e-6]:
        print 'eps =', eps
        print approx_hess(xk,fun2,eps,args)[0] - 2*np.dot(x.T, x)

    hcs2 = approx_hess_cs2(xk,fun2,args=args)
    print 'hcs2'
    print hcs2 - 2*np.dot(x.T, x)

    hfd3 = approx_hess3(xk,fun2,args=args)
    print 'hfd3'
    print hfd3 - 2*np.dot(x.T, x)

    import numdifftools as nd
    hnd = nd.Hessian(lambda a: fun2(a, y, x))
    hessnd = hnd(xk)
    print 'numdiff'
    print hessnd - 2*np.dot(x.T, x)
    #assert_almost_equal(hessnd, he[0])
    gnd = nd.Gradient(lambda a: fun2(a, y, x))
    gradnd = gnd(xk)
'''
>>> hnd = nd.Hessian(lambda a: fun2(a, x))
>>> hnd(xk)
array([[ 216.87702746,   -3.41892545,    1.87887281],
       [  -3.41892545,  180.76379116,  -13.74326021],
       [   1.87887281,  -13.74326021,  198.5155617 ]])
>>> he
(array([[ 216.87702746,   -3.41892545,    1.87887281],
       [  -3.41892545,  180.76379116,  -13.74326021],
       [   1.87887281,  -13.74326021,  198.5155617 ]]), array([ 2.35204474,  1.92684939,  2.11920745]))
>>> hnd = nd.Gradient(lambda a: fun2(a, x))
>>> hnd(xk)
array([  0.00000000e+00,   1.40036521e-14,  -2.59014117e-14])
>>> hnd((1.2, 1.2, 1.2))
array([ 41.58106741,  34.51076432,  38.95952468])
>>> approx_fprime1(xk,fun1,epsilon,*args).sum(0)
array([  1.08438310e-04,   9.03821177e-05,   9.92578403e-05])
>>> approx_fprime1((1.2, 1.2, 1.2),fun1,epsilon,*args).sum(0)
array([ 41.58117585,  34.5108547 ,  38.95962393])
>>> approx_fprime((1.2, 1.2, 1.2),fun2,epsilon,*args).sum(0)
115.05165448078003
>>> approx_fprime((1.2, 1.2, 1.2),fun2,epsilon,*args)
array([ 41.58117584,  34.5108547 ,  38.95962393])
>>> epsilon
9.9999999999999995e-007
>>> approx_hess(np.array([1.2, 1.2, 1.2]),fun2,epsilon,*args)
(3, 3) H= [[  1.00000000e-12   1.00000000e-12   1.00000000e-12]
 [  1.00000000e-12   1.00000000e-12   1.00000000e-12]
 [  1.00000000e-12   1.00000000e-12   1.00000000e-12]]
h= [[  1.00000000e-12   1.00000000e-12   1.00000000e-12]
 [  1.00000000e-12   1.00000000e-12   1.00000000e-12]
 [  1.00000000e-12   1.00000000e-12   1.00000000e-12]]
(array([[ 216.87718291,   -3.41771056,    1.87938554],
       [  -3.41771056,  180.76384836,  -13.74189651],
       [   1.87938554,  -13.74189651,  198.51498226]]), array([ 41.58117585,  34.51085471,  38.95962394]))


>>> hnd = nd.Hessian(lambda a: fun2(a, x))
>>> hnd((1.2, 1.2, 1.2))
array([[ 216.87702746,   -3.41892545,    1.87887281],
       [  -3.41892545,  180.76379116,  -13.74326021],
       [   1.87887281,  -13.74326021,  198.5155617 ]])
>>> hnd(xk)
array([[ 216.87702746,   -3.41892545,    1.87887281],
       [  -3.41892545,  180.76379116,  -13.74326021],
       [   1.87887281,  -13.74326021,  198.5155617 ]])
>>> j = approx_fprime1((1.2, 1.2, 1.2),fun1,epsilon,*args)
>>> np.dot(j.T,j)
array([[ 90.37418663,  23.81355855,  33.2936759 ],
       [ 23.81355855,  49.01569714,  16.70507137],
       [ 33.2936759 ,  16.70507137,  82.24708274]])


>>> heb = approx_hess(xk,fun2,-1e-6,*args)
(3, 3) H= [[  1.00000000e-12   1.00000000e-12   1.00000000e-12]
 [  1.00000000e-12   1.00000000e-12   1.00000000e-12]
 [  1.00000000e-12   1.00000000e-12   1.00000000e-12]]
h= [[  1.00000000e-12   1.00000000e-12   1.00000000e-12]
 [  1.00000000e-12   1.00000000e-12   1.00000000e-12]
 [  1.00000000e-12   1.00000000e-12   1.00000000e-12]]
>>> hef
(array([[ 216.87707189,   -3.41926487,    1.87860838],
       [  -3.41926487,  180.76329321,  -13.74345082],
       [   1.87860838,  -13.74345082,  198.51542631]]), array([  1.08438369e-04,   9.03821462e-05,   9.92579352e-05]))
>>> heb
(array([[ 216.87729394,   -3.41848772,    1.87905247],
       [  -3.41848772,  180.76384832,  -13.7433398 ],
       [   1.87905247,  -13.7433398 ,  198.51575938]]), array([ -1.08438258e-04,  -9.03819242e-05,  -9.92578242e-05]))
>>> (hef[1]+heb[1])/2.
array([  5.55111512e-11,   1.11022302e-10,   5.55111512e-11])
>>> (hef[0]+heb[0])/2.
array([[ 216.87718291,   -3.41887629,    1.87883042],
       [  -3.41887629,  180.76357077,  -13.74339531],
       [   1.87883042,  -13.74339531,  198.51559284]])
>>> hnd = nd.Hessian(lambda a: fun2(a, x))
>>> hnd(xk)
array([[ 216.87702746,   -3.41892545,    1.87887281],
       [  -3.41892545,  180.76379116,  -13.74326021],
       [   1.87887281,  -13.74326021,  198.5155617 ]])
>>> j = approx_fprime1((1.2, 1.2, 1.2),fun1,epsilon,*args)
>>> np.dot(j.T,j)
array([[ 90.37418663,  23.81355855,  33.2936759 ],
       [ 23.81355855,  49.01569714,  16.70507137],
       [ 33.2936759 ,  16.70507137,  82.24708274]])
'''
