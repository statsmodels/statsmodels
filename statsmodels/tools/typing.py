from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from pandas import DataFrame, Series

ArrayLike1D = Union[Sequence[Union[float, int]], np.ndarray, Series]
ArrayLike2D = Union[
    Sequence[Sequence[Union[float, int]]], np.ndarray, DataFrame
]
ArrayLike = Union[ArrayLike1D, ArrayLike2D]
