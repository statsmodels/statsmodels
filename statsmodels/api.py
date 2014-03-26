import statsmodels.iolib as iolib
import statsmodels.datasets as datasets
import statsmodels.tools as tools
from statsmodels.tools.tools import add_constant, categorical
import statsmodels.regression as regression
from statsmodels.regression.linear_model import OLS, GLS, WLS, GLSAR
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families
import statsmodels.robust as robust
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.discrete.discrete_model import (Poisson, Logit, Probit,
                                                 MNLogit, NegativeBinomial)
from statsmodels.tsa import api as tsa
from statsmodels.nonparametric import api as nonparametric
import statsmodels.distributions as distributions
from statsmodels.__init__ import test
from statsmodels import version
from statsmodels.info import __doc__
from statsmodels.graphics.gofplots import qqplot, qqplot_2samples, qqline, ProbPlot
from statsmodels.graphics import api as graphics
from statsmodels.stats import api as stats
from statsmodels.emplike import api as emplike

from statsmodels.formula import api as formula

from statsmodels.iolib.smpickle import load_pickle as load

from statsmodels.tools.print_version import show_versions

import os

chmpath = os.path.join(os.path.dirname(__file__), 'statsmodelsdoc.chm')
if os.path.exists(chmpath):
    def open_help(chmpath=chmpath):
        from subprocess import Popen
        p = Popen(chmpath, shell=True)

del os
del chmpath