# -*- coding: utf-8 -*-
from libc.math cimport exp, log
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime, approx_hess1
from cython cimport boundscheck
from cython.parallel import prange
import matplotlib.pyplot as pp
from pandas import DataFrame
#
#function to bound parameter between 0 and 1
#
cdef double bound(double param) nogil: 
    return exp(param)/(1.0+exp(param))
#
#derivative of the function used in VAR-matrix computation
#
cdef double bound_derivative(double param) nogil: 
    return exp(param)/((1.0+exp(param))**2)
#
#Weibull density function with absorbing state, prop is the proportion of 
#population in absorbing state. Exponential reparemeterization.
#
cdef inline double weibull_as_density(double t, double k, double lam, double prop) nogil: 
    return (1.0-bound(prop))*(exp(k)/exp(lam))*((t/exp(lam))**(exp(k)-1.0))*exp(-(t/exp(lam))**exp(k))
#Same for survival function  
cdef inline double weibull_as_survival(double t, double k, double lam, double prop) nogil:
    return bound(prop)+(1.0-bound(prop))*exp(-(t/exp(lam))**exp(k))
#
#Due to a problem with  elif\else inside prange loops when building this package, survival and 
#density are joined in a function
cdef inline double individual_likelihood(double t, double full, double k, double lam, double prop) nogil :
    if full :
        return weibull_as_density(t, k, lam, prop)
    else :
        return weibull_as_survival(t, k, lam, prop)
#
#
#Survival function unbound for graph
#
cdef double weibull_as_survival_graph(double t, double k, double lam, double prop) nogil:
    return prop+(1.0-prop)*exp(-(t/lam)**k)
#
#Func to transform array to memory view
#
def _converter(array) :
    """function returns buffer for fast 1-d array access"""
    cdef double[:] view
    if (type(array) != np.ndarray) & (array.dtype != "float64") :
        raise TypeError("only numpy arrays")
    else :
        view = array 
        return view[:]
#
#Likelihood function on the whole sample
#
@boundscheck(False)
cdef double likelihood_func(double[:] params, double[:] duration, double[:] is_complete) nogil:
    cdef:
        int i
        double LL = 0.0
        int I = duration.shape[0]
    for i in prange(I, schedule = "static", chunksize = 5000) :
        LL += log(individual_likelihood(duration[i], is_complete[i], params[0], params[1], params[2]))
    return LL            
    
#
#Class for estimation
#
class Weibull_AS_Estimator : 
    def __init__(self, duration, is_complete, dataset=None) :
        if (dataset is not None) & (type(duration) is str) & (type(is_complete) is str) :
            self.duration = dataset[duration].astype(float).values
            self.complete = dataset[is_complete].astype(float).values
            self._duration_view = _converter(self.duration)
            self._complete_view = _converter(self.complete)
        elif (dataset is None) & (type(duration) is np.array) & (type(is_complete) is np.array) :
            self.duration = np.array(duration, dtype = float)
            self.complete = np.array(is_complete, dtype = float)
            self._duration_view = _converter(self.duration)
            self._complete_view = _converter(self.complete)
        else :
            raise TypeError("Duration and is complete must be name of columns of a dataset or Numpy Arrays")
        self._estimated = False
    def _for_estimation(self, params) :
        return -likelihood_func(_converter(params), self._duration_view, self._complete_view)
    def fit(self, tol = 1e-9) :
        self.result = minimize(self._for_estimation, np.zeros(shape = 3), method = "Powell", tol = tol) 
        self.estimated_coeff = {"k" : exp(self.result.x[0]),
                                "lambda" : exp(self.result.x[1]),
                                "AS prop" : bound(self.result.x[2])}
        self.neg_hessian_inv = np.linalg.inv(approx_hess1(self.result.x, self._for_estimation))
        grad = np.array([[exp(self.result.x[0]), 0.0, 0.0], [0.0, exp(self.result.x[1]), 0.0],
                          [0.0, 0.0, bound_derivative(self.result.x[2])]])
        estimated_var_mat = grad.dot(self.neg_hessian_inv).dot(grad.T)
        self.estimated_coeff_var = {"k var" : estimated_var_mat[0, 0],
                                    "lambda var" : estimated_var_mat[1, 1],
                                    "AS var" : estimated_var_mat[2, 2]}
        self._estimated = True
    def plot_survival(self, x_max = None, figsize = (10, 5)) :
        if not self._estimated :
            print("Model must be fitted first")
        else :
            if x_max is not None :
                X = np.linspace(start = 0, stop = x_max, num = 1500)
            else :
                #finding where survival reach it's lowest point
                x_max = 1.0
                while (weibull_as_survival_graph(x_max, self.estimated_coeff["k"], self.estimated_coeff["lambda"],
                                              self.estimated_coeff["AS prop"]) > (self.estimated_coeff["AS prop"] + 5e-5)) :
                    x_max += 1.0
                X = np.linspace(start = 0, stop = x_max, num = 1500)
            y = np.array([weibull_as_survival_graph(x, self.estimated_coeff["k"], self.estimated_coeff["lambda"],
                                              self.estimated_coeff["AS prop"]) for x in X])
            pp.style.use("fivethirtyeight")
            pp.figure(figsize = figsize)
            pp.plot(X, y)
            pp.xlim([-0.5, x_max])
            pp.ylim([0.0, 1.0])
            
def test() :
    test_sample = np.random.exponential(scale = 2, size = 25000)
    test_sample[20000:] = 10000.0
    test_is_complete = np.ones(shape = 25000, dtype = float)
    test_is_complete[20000:] = 0.0
    test_df = DataFrame({"duration" : test_sample, "is_complete" : test_is_complete})
    test_reg = Weibull_AS_Estimator("duration", "is_complete", test_df)
    test_reg.fit(tol = 0.0)
    estimated_coefficients = np.array([test_reg.estimated_coeff["lambda"],
                                      test_reg.estimated_coeff["k"],
                                      test_reg.estimated_coeff["AS prop"]])
    true_coefficients = np.array([2.0, 1.0, 0.2])
    np.testing.assert_allclose(estimated_coefficients, true_coefficients, rtol = 0.05)
    print("OK")    