"""
Test Results for discrete models from R (mlogit package)
#TODO Cross check with Biogeme: http://biogeme.epfl.ch/

"""
import os
import numpy as np

cur_dir = os.path.abspath(os.path.dirname(__file__))


class Modechoice():
    """
    # R code

    library("mlogit", "TravelMode")
    names(TravelMode)<- c("individual", "mode", "choice", "ttme", "invc",
                             "invt", "gc", "hinc", "psize")
    TravelMode$hinc_air <- with(TravelMode, hinc * (mode == "air"))
    res <- mlogit(choice ~ gc + ttme + hinc_air, data = TravelMode,
                shape = "long", alt.var = "mode", reflevel = "car")
    summary(res)
    model$hessian       #the hessian of the log-likelihood at convergence

    """
    def __init__(self):
        self.nobs = 210

    def CLogit(self):
        self.params = [-0.01550151, -0.09612462, 5.20743293, 0.013287011,
                       3.16319033, 3.86903570]
