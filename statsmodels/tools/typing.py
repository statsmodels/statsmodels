"""Typing aliases used by statsmodels"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import (
        ArrayLike,
        DTypeLike,
        NDArray,
        _FloatLike_co,
        _UIntLike_co,
    )
    from pandas import DataFrame, Series

    _ExtendedFloatLike_co = _FloatLike_co | _UIntLike_co
    NumericArray = NDArray[Any, np.dtype[_ExtendedFloatLike_co]]
    Float64Array = NDArray[Any, np.double]
    ArrayLike1D = Sequence[float | int] | NumericArray | Series
    ArrayLike2D = Sequence[Sequence[float | int]] | NumericArray | DataFrame
else:
    ArrayLike = Any
    DTypeLike = Any
    Float64Array = Any
    NumericArray = Any
    ArrayLike1D = Any
    ArrayLike2D = Any
    NDArray = Any

__all__ = [
    "ArrayLike",
    "ArrayLike1D",
    "ArrayLike2D",
    "DTypeLike",
    "Float64Array",
    "NDArray",
    "NumericArray",
]
