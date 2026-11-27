"""Typing aliases used by statsmodels"""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from pandas import DataFrame, Series

NumericArray: TypeAlias = NDArray[np.bool_ | np.integer | np.floating]
Float64Array: TypeAlias = NDArray[np.float64]
ArrayLike1D: TypeAlias = Sequence[float] | NumericArray | Series
ArrayLike2D: TypeAlias = Sequence[Sequence[float]] | NumericArray | DataFrame

__all__ = [
    "ArrayLike",
    "ArrayLike1D",
    "ArrayLike2D",
    "DTypeLike",
    "Float64Array",
    "NDArray",
    "NumericArray",
]
