from __future__ import annotations

from typing import Any, Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd

try:
    from patsy.missing import NAAction

    DEFAULT_ENGINE = "patsy"

    class NAAction(NAAction):
        # monkey-patch so we can handle missing values in 'extra' arrays later
        def _handle_NA_drop(self, values, is_NAs, origins):
            total_mask = np.zeros(is_NAs[0].shape[0], dtype=bool)
            for is_NA in is_NAs:
                total_mask |= is_NA
            good_mask = ~total_mask
            self.missing_mask = total_mask
            # "..." to handle 1- versus 2-dim indexing
            return [v[good_mask, ...] for v in values]

except ImportError:
    DEFAULT_ENGINE = "formulaic"

    class NAAction:
        def __init__(self, on_NA="", NA_types=("",)):
            pass


class _FormulaOption:
    def __init__(self, default_engine: Literal["patsy", "formulaic"] | None = None):
        if default_engine is None:
            default_engine = DEFAULT_ENGINE

        self._formula_engine = default_engine
        self._allowed_options = ("patsy", "formulaic")

    @property
    def formula_engine(self) -> Literal["patsy", "formulaic"]:
        return self._formula_engine

    @formula_engine.setter
    def formula_engine(self, value: Literal["patsy", "formulaic"]) -> None:
        if value not in self._allowed_options:
            msg = "Invalid formula engine option. Must be "
            if len(self._allowed_options) == 1:
                msg += f"{self._allowed_options[0]}"
            else:
                allowed = list(self._allowed_options)
                allowed[-1] = f"or {allowed[-1]}"
                if len(allowed) > 2:
                    msg += ", ".join(allowed)
                else:
                    msg += " ".join(allowed)
            raise ValueError(f"{msg}.")

        self._formula_engine = value


class _Default:
    def __init__(self, name=""):
        self._name = ""

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name


_NoDefault = _Default("<no default value>")


class LinearConstraintValues(NamedTuple):
    constraint_matrix: np.ndarray
    constraint_values: np.ndarray
    variable_names: list[str]


class FormulaManager:
    def __init__(self, engine: Literal["patsy", "formulaic"] | None = None):
        self._engine = self._get_engine(engine)
        self._spec = None

    def _get_engine(
        self, engine: Literal["patsy", "formulaic"] | None = None
    ) -> Literal["patsy", "formulaic"]:
        # Patsy for now, to be changed to a user-settable variable before release
        _engine: Literal["patsy", "formulaic"]

        if engine is not None:
            _engine = engine
        else:
            import statsmodels.formula

            _engine = statsmodels.formula.options.formula_engine

        assert _engine is not None
        if _engine not in ("patsy", "formulaic"):
            raise ValueError(
                f"Unknown engine: {_engine}. Only patsy and formulaic are supported."
            )
        return _engine

    @property
    def engine(self):
        return self._engine

    @property
    def spec(self):
        return self._spec

    def get_arrays(
        self,
        formula,
        data,
        eval_env=0,
        pandas=True,
        attach_spec=True,
        na_action=None,
    ) -> (
        np.ndarray
        | tuple[np.ndarray, np.ndarray]
        | pd.DataFrame
        | tuple[pd.DataFrame, pd.DataFrame]
    ):
        if isinstance(eval_env, (int, np.integer)):
            eval_env = int(eval_env) + 1
        if self._engine == "patsy":
            import patsy

            return_type = "dataframe" if pandas else "matrix"
            kwargs = {}
            if na_action:
                kwargs["NA_action"] = na_action
            if (
                isinstance(
                    formula, (patsy.design_info.DesignInfo, patsy.desc.ModelDesc)
                )
                or "~" not in formula
                or formula.strip().startswith("~")
            ):
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

        else:  # self._engine == "formulaic":
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

    def get_linear_constraints(
        self, constraints: np.ndarray | str | Sequence[str], variable_names: list[str]
    ):
        if self._engine == "patsy":
            from patsy.design_info import DesignInfo

            lc = DesignInfo(variable_names).linear_constraint(constraints)
            return LinearConstraintValues(
                constraint_matrix=lc.coefs,
                constraint_values=lc.constants,
                variable_names=lc.variable_names,
            )
        else:  # self._engine == "formulaic"
            import formulaic.utils.constraints

            values = np.zeros(constraints.shape[0])
            lc_f = formulaic.utils.constraints.LinearConstraints(
                constraints, values, variable_names=variable_names
            )
            return LinearConstraintValues(
                constraint_matrix=lc_f.constraint_matrix,
                constraint_values=lc_f.constraint_values,
                variable_names=lc_f.variable_names,
            )

    def get_empty_eval_env(self):
        if self._engine == "patsy":
            from patsy.eval import EvalEnvironment

            return EvalEnvironment({})
        else:
            return {}

    def remove_intercept(self, terms):
        """
        Remove intercept from Patsy terms.
        """
        if self._engine == "patsy":
            from patsy.desc import INTERCEPT

            if INTERCEPT in terms:
                terms.remove(INTERCEPT)
            return terms
        else:
            raise NotImplementedError(
                "Removing intercept is not implemented for formulaic engine."
            )

    def has_intercept(self, spec):
        if self._engine == "patsy":
            from patsy.desc import INTERCEPT

            return INTERCEPT in spec.terms
        else:
            raise NotImplementedError(
                "Checking for intercept is not implemented for formulaic engine."
            )

    def intercept_idx(self, spec):
        """
        Returns boolean array index indicating which column holds the intercept.
        """
        from numpy import array

        if self._engine == "patsy":
            from patsy.desc import INTERCEPT

            return array([INTERCEPT == i for i in spec.terms])
        else:
            raise NotImplementedError(
                "Intercpet idx has not been implemented for formulaic"
            )

    def get_na_action(self, action: str = "drop", types: Sequence[Any] = _NoDefault):
        types = ["None", "NaN"] if types is _NoDefault else types
        if self._engine == "patsy":
            return NAAction(on_NA=action, NA_types=types)
        else:
            return action

    def get_spec(self, formula):
        if self._engine == "patsy":
            import patsy

            return patsy.ModelDesc.from_formula(formula)
        else:
            import formulaic

            return formulaic.ModelSpec(formula)

    def get_column_names(self, frame):
        return self.get_model_spec(frame).column_names

    def get_term_name_slices(self, frame):
        return self.get_model_spec(frame).term_name_slices

    def get_model_spec(self, frame, optional=False):
        if self._engine == "patsy":
            if optional and not hasattr(frame, "design_info"):
                return None
            return frame.design_info
        else:
            if optional and not hasattr(frame, "model_spec"):
                return None
            return frame.model_spec