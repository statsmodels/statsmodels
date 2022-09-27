# -*- coding: utf-8 -*-
from Weibull_AS_cytools import _converter, generate_plot_values, py_likelihood_func
import numpy as np
import matplotlib.pyplot as pp
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess1
from pandas import DataFrame
from math import exp
#
#function to bound parameter between 0 and 1
#
def bound(param) : 
    return exp(param)/(1.0+exp(param))
#
#derivative of transformation to bound parameter between 0 and 1
#used in VAR-matrix computation
#
def bound_derivative(param) : 
    return exp(param)/((1.0+exp(param))**2)
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
        return -py_likelihood_func(_converter(params), self._duration_view, self._complete_view)
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
    def plot_survival(self, x_max = None, figsize = (10, 5), nb_dots = 1000) :
        if not self._estimated :
            print("Model must be fitted first")
        else :
            X, y = generate_plot_values(self.estimated_coeff, nb_dots, x_max)
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