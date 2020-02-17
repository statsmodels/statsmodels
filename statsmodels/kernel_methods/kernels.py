r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module providing a set of kernels for use within the kernel_methods package.
"""

from ._kernels import *
from ._kernels1d import TriCube, Epanechnikov, EpanechnikovOrder4, GaussianOrder4
# from ._kernelsnd import Gaussian
from ._kernelsnc import AitchisonAitken, WangRyzin
""" List of 1D kernels """
kernels1D = [
    Gaussian1D, TriCube, Epanechnikov, EpanechnikovOrder4, GaussianOrder4
]
""" List of nD kernels """
kernelsnD = [Gaussian]
""" List of non-continuous kernels """
kernelsNC = [AitchisonAitken, WangRyzin]
