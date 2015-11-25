"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains all the methods for computing the KDE.
"""

from __future__ import division, absolute_import, print_function

from ._kde_methods import KDEMethod, filter_exog  # noqa

from ._kde1d_methods import KDE1DMethod  # noqa
from ._kde1d_linear_combination import LinearCombination  # noqa
from ._kde1d_cyclic import Cyclic1D  # noqa
from ._kde1d_reflection import Reflection1D  # noqa
from ._kde1d_renormalization import Renormalization  # noqa
from ._kde1d_transform import Transform1D  # noqa
from ._kde_multivariate import Multivariate  # noqa
from ._kdenc_methods import Unordered, Ordered  # noqa
from ._kdend_methods import KDEnDMethod, Cyclic, Reflection  # noqa
from ._kde1d_transform import LogTransform, ExpTransform, Transform, create_transform  # noqa
from ._kde1d_methods import convolve, generate_grid1d  # noqa
from ._kdend_methods import generate_grid  # noqa
