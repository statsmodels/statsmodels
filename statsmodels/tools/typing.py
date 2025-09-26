from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from collections.abc import Sequence
    from packaging.version import parse

    import numpy as np

    if parse(np.__version__) < parse("1.22.0"):
        raise NotImplementedError(
            "NumPy 1.22.0 or later required for type checking"
        )
    from numpy.typing import (
        ArrayLike as ArrayLike,  # noqa: PLC0414
        DTypeLike,
        NDArray,
        _FloatLike_co,
        _UIntLike_co,
    )
    from pandas import DataFrame, Series

    _ExtendedFloatLike_co = Union[_FloatLike_co, _UIntLike_co]
    NumericArray = NDArray[Any, np.dtype[_ExtendedFloatLike_co]]
    Float64Array = NDArray[Any, np.double]
    ArrayLike1D = Union[Sequence[Union[float, int]], NumericArray, Series]
    ArrayLike2D = Union[
        Sequence[Sequence[Union[float, int]]], NumericArray, DataFrame
    ]
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
    "DTypeLike",
    "Float64Array",
    "ArrayLike1D",
    "ArrayLike2D",
    "NDArray",
    "NumericArray",
]
