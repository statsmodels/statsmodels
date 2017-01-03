# -*- coding: utf-8 -*-
from libc.math cimport exp, log
import numpy as np
from cython cimport boundscheck
from cython.parallel import prange
#
#function to bound parameter between 0 and 1
#
cdef inline double bound(double param) nogil: 
    return exp(param)/(1.0+exp(param))
#
#Weibull density function with absorbing state, prop is the proportion of 
#population in absorbing state. Exponential reparemeterization.
#
cdef inline double weibull_as_density(double t, double k, double lam, double prop) nogil: 
    return (1.0-prop)*(k/lam)*((t/lam)**(k-1.0))*exp(-(t/lam)**k)
#Same for survival function  
cdef inline double weibull_as_survival(double t, double k, double lam, double prop) nogil:
    return prop+(1.0-prop)*exp(-(t/lam)**k)
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
        double exp_k = exp(params[0])
        double exp_lam = exp(params[1])
        double bound_prop = bound(params[2])
        int i
        double LL = 0.0
        int I = duration.shape[0]
    for i in prange(I, schedule = "static", chunksize = 5000) :
        LL += log(individual_likelihood(duration[i], is_complete[i], exp_k, exp_lam, bound_prop))
    return LL            
#
#Python interfacing function
#
def py_likelihood_func(params_view, duration_view, complete_view) : 
    return likelihood_func(params_view, duration_view, complete_view)
#
#Function used to create values to plot
#   
def generate_plot_values(estimated_coeff, nb_dots, x_max = None) :
    if x_max is None :
        x_max = 1.0
        while (weibull_as_survival_graph(x_max,
                estimated_coeff["k"], estimated_coeff["lambda"],
                estimated_coeff["AS prop"]) > (estimated_coeff["AS prop"] + 5e-5)) :
                        x_max += 1.0
    X = np.linspace(start = 0, stop = x_max, num = 1500)
    result = np.array([weibull_as_survival_graph(x, estimated_coeff["k"], estimated_coeff["lambda"],
                                              estimated_coeff["AS prop"]) for x in X])
    return (X, result)