from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


class FormulaManager:
    def __init__(self, engine: Literal["patsy", "formulaic"] | None = None):
        self._engine = self._get_engine(engine)
        self._spec = None

    def _get_engine(self, engine: Literal["patsy", "formulaic"] | None = None) -> str:
        # Patsy for now, to be changed to a user-settable variable before release
        engine = engine or "patsy"

        if engine not in ("patsy", "formulaic"):
            raise ValueError(
                f"Unknown engine: {engine}. Only "
                "patsy"
                " and "
                "formulaic"
                " "
                "are supported."
            )
        return engine

    @property
    def spec(self):
        return self._spec

    def get_arrays(
        self, formula, data, eval_env=0, pandas=True, attach_spec=True, na_action=None,
    ) -> (
        np.ndarray
        | tuple[np.ndarray, np.ndarray]
        | pd.DataFrame
        | tuple[pd.DataFrame, pd.DataFrame]
    ):
        try:
            eval_env = int(eval_env) + 1
        except Exception:
            pass
        if self._engine == "patsy":
            import patsy

            return_type = "dataframe" if pandas else "matrix"
            kwargs ={}
            if na_action:
                kwargs["NA_action"] = na_action
            if isinstance(formula, patsy.design_info.DesignInfo) or "~" not in formula:
                output = patsy.dmatrix(
                    formula, data, eval_env=eval_env, return_type=return_type, **kwargs
                )
            else:  # "~" in formula:
                output = patsy.dmatrices(
                    formula, data, eval_env=eval_env, return_type=return_type, **kwargs
                )
            if attach_spec:
                if isinstance(output, tuple):
                    self._spec = output[1].design_info
                else:
                    self._spec = output.design_info
            return output

        elif self._engine == "formulaic":
            import formulaic

            kwargs = {}
            if pandas:
                kwargs["output"] = "pandas"
            if na_action:
                kwargs["na_action"] = na_action
            output = formulaic.model_matrix(formula, data, context=eval_env, **kwargs)
            if attach_spec:
                if hasattr(output, "rhs"):
                    self._spec = output.rhs.model_spec
                else:
                    self._spec = output.model_spec
            return output
