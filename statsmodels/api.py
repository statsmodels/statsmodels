from . import iolib
from . import datasets
from . import tools
from .tools.tools import add_constant, categorical
from . import regression
from .regression.linear_model import OLS, GLS, WLS, GLSAR
from .regression.recursive_ls import RecursiveLS
from .regression.quantile_regression import QuantReg
from .regression.mixed_linear_model import MixedLM
from .genmod import api as genmod
from .genmod.api import GLM, GEE, OrdinalGEE, NominalGEE, families, cov_struct
from . import robust
from .robust.robust_linear_model import RLM
from .discrete.discrete_model import (Poisson, Logit, Probit,
                                      MNLogit, NegativeBinomial)
from .tsa import api as tsa
from .duration.survfunc import SurvfuncRight
from .duration.hazard_regression import PHReg
from .imputation.mice import MICE, MICEData
from .nonparametric import api as nonparametric
from . import distributions
from .__init__ import test
from . import version
from .info import __doc__
from .graphics.gofplots import qqplot, qqplot_2samples, qqline, ProbPlot
from .graphics import api as graphics
from .stats import api as stats
from .emplike import api as emplike
from .duration import api as duration
from .multivariate.pca import PCA
from .multivariate.manova import MANOVA

from .formula import api as formula

from .iolib.smpickle import load_pickle as load

from .tools.print_version import show_versions
from .tools.web import webdoc

import os

chmpath = os.path.join(os.path.dirname(__file__), 'statsmodelsdoc.chm')
if os.path.exists(chmpath):
    def open_help(chmpath=chmpath):
        from subprocess import Popen

        p = Popen(chmpath, shell=True)

del os
del chmpath
