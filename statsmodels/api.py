# -*- coding: utf-8 -*-
# flake8: noqa

__all__ = [
    "iolib",
    "datasets",
    "tools",
    "add_constant",
    "categorical",
    "regression",
    "OLS",
    "GLS",
    "WLS",
    "GLSAR",
    "RecursiveLS",
    "QuantReg",
    "MixedLM",
    "genmod",
    "GLM",
    "GEE",
    "OrdinalGEE",
    "NominalGEE",
    "families",
    "cov_struct",
    "BinomialBayesMixedGLM",
    "PoissonBayesMixedGLM",
    "robust",
    "RLM",
    "Poisson",
    "Logit",
    "Probit",
    "MNLogit",
    "NegativeBinomial",
    "GeneralizedPoisson",
    "NegativeBinomialP",
    "ZeroInflatedNegativeBinomialP",
    "ZeroInflatedPoisson",
    "ZeroInflatedGeneralizedPoisson",
    "tsa",
    "SurvfuncRight",
    "PHReg",
    "MICE",
    "MICEData",
    "BayesGaussMI",
    "MI",
    "nonparametric",
    "distributions",
    "test",
    "GLMGam",
    "gam",
    "show_versions",
    "webdoc",
    "qqplot",
    "stats",
    "graphics",
    "emplike",
    "PCA",
    "MANOVA",
    "formula",
    "multivariate",
    "Factor",
    "qqplot_2samples",
    "qqline",
    "ProbPlot",
    "duration",
    "__version__",
]


from . import datasets, distributions, iolib, regression, robust, tools
from .__init__ import test
from .discrete.count_model import (
    ZeroInflatedGeneralizedPoisson,
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedPoisson,
)
from .discrete.discrete_model import (
    GeneralizedPoisson,
    Logit,
    MNLogit,
    NegativeBinomial,
    NegativeBinomialP,
    Poisson,
    Probit,
)
from .duration import api as duration
from .duration.hazard_regression import PHReg
from .duration.survfunc import SurvfuncRight
from .emplike import api as emplike
from .formula import api as formula
from .gam import api as gam
from .gam.generalized_additive_model import GLMGam
from .genmod import api as genmod
from .genmod.api import (
    GEE,
    GLM,
    BinomialBayesMixedGLM,
    NominalGEE,
    OrdinalGEE,
    PoissonBayesMixedGLM,
    cov_struct,
    families,
)
from .graphics import api as graphics
from .graphics.gofplots import ProbPlot, qqline, qqplot, qqplot_2samples
from .imputation.bayes_mi import MI, BayesGaussMI
from .imputation.mice import MICE, MICEData
from .iolib.smpickle import load_pickle
from .multivariate import api as multivariate
from .multivariate.factor import Factor
from .multivariate.manova import MANOVA
from .multivariate.pca import PCA
from .nonparametric import api as nonparametric
from .regression.linear_model import GLS, GLSAR, OLS, WLS
from .regression.mixed_linear_model import MixedLM
from .regression.quantile_regression import QuantReg
from .regression.recursive_ls import RecursiveLS
from .robust.robust_linear_model import RLM
from .stats import api as stats
from .tools.print_version import show_versions
from .tools.tools import add_constant, categorical
from .tools.web import webdoc
from .tsa import api as tsa

load = load_pickle

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
