import iolib, datasets, tools
from tools.tools import add_constant, categorical
import regression
from .regression.linear_model import OLS, GLS, WLS, GLSAR
from .genmod.generalized_linear_model import GLM
from .genmod import families
import robust
from .robust.robust_linear_model import RLM
from .discrete.discrete_model import Poisson, Logit, Probit, MNLogit
from .tsa import api as tsa
import nonparametric
from __init__ import test
import version
from info import __doc__

import os

chmpath = os.path.join(os.path.dirname(__file__),
                       'docs\\build\\htmlhelp\\statsmodelsdoc.chm')
if os.path.exists(chmpath):
    def open_help(chmpath=chmpath):
        from subprocess import Popen
        p = Popen(chmpath, shell=True)


del os
del chmpath



