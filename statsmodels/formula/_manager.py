from __future__ import annotations

from collections import defaultdict
import os
from typing import Any, Literal, Mapping, NamedTuple, Sequence
import warnings

import numpy as np
import pandas as pd

HAVE_PATSY = False
HAVE_FORMULAIC = False

DEFAULT_FORMULA_ENGINE = os.environ.get("SM_FORMULA_ENGINE", None)
if DEFAULT_FORMULA_ENGINE not in ("formulaic", "patsy", None):
    raise ValueError(f"Invalid value for SM_FORMULA_ENGINE: {DEFAULT_FORMULA_ENGINE}")

try:
    import patsy
    import patsy.missing

    DEFAULT_FORMULA_ENGINE = DEFAULT_FORMULA_ENGINE or "patsy"

    class NAAction(patsy.missing.NAAction):
        # monkey-patch so we can handle missing values in 'extra' arrays later
        def _handle_NA_drop(self, values, is_nas, origins):
            total_mask = np.zeros(is_nas[0].shape[0], dtype=bool)
            for is_NA in is_nas:
                total_mask |= is_NA
            good_mask = ~total_mask
            self.missing_mask = total_mask
            # "..." to handle 1- versus 2-dim indexing
            return [v[good_mask, ...] for v in values]

    HAVE_PATSY = True

except ImportError:
    DEFAULT_FORMULA_ENGINE = DEFAULT_FORMULA_ENGINE or "formulaic"

    class NAAction:
        def __init__(self, on_na="", na_types=("",)):
            pass


try:
    import formulaic  # noqa: F401
    import formulaic.parser
    import formulaic.utils.constraints

    HAVE_FORMULAIC = True
except ImportError:
    # Nothing to do if formulaic is not available
    pass


EVAL_ENV_WARNING = """\
EvalEnvironment is deprecated and support be removed in a future version. You can
pass variables using a dict[str, value] where str is the variable name and the
value is the array of values.
"""


def _check_data(data):
    """
    Check if data is a DataFrame and issue a warning if it is not.

    Parameters
    ----------
    data : {dict, list, recarray, DataFrame}
        The data used to create a formula.

    Notes
    -----
    Warns if data is not a DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        warnings.warn(
            f"Using {type(data).__name__} data structures with formula is "
            "deprecated and will be removed in a future version of statsmodels. "
            "DataFrames are the only supported data structure.",
            DeprecationWarning,
        )


class _FormulaOption:
    def __init__(self, default_engine: Literal["patsy", "formulaic"] | None = None):
        if default_engine is None:
            default_engine = DEFAULT_FORMULA_ENGINE

        self._formula_engine = default_engine
        self._allowed_options = tuple()
        if HAVE_PATSY:
            self._allowed_options += ("patsy",)
        if HAVE_FORMULAIC:
            self._allowed_options += ("formulaic",)
        self.ordering = "legacy"

    @property
    def formula_engine(self) -> Literal["patsy", "formulaic"]:
        """
        Get or set the formula engine

        Returns
        -------
        str: {"patsy", "formulaic"}
            The name of the formula engine.
        """
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

    @property
    def ordering(self):
        """
        Get or set the ordering.

        Returns
        -------
        {"degree", "sort", "none", "legacy"}
            The name of the ordering. Only used if engine is formulaic.
        """
        return self._ordering

    @ordering.setter
    def ordering(self, value: str):
        if value not in ("degree", "sort", "none", "legacy"):
            raise ValueError(
                "Invalid ordering option. Must be 'degree', 'sort', 'none', or 'legacy."
            )
        self._ordering = value


class _Default:
    def __init__(self, name=""):
        self._name = name

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
    """
    Abstraction class that provides a common interface to patsy and formulaic.

    Designed to aid in the transition from patsy to formulaic.

    Parameters
    ----------
    engine : {"patsy", "formulaic"} or None
        The formula engine to use. If None, the default engine, which appears
        in the attribute statsmodels.formula.formula_options.engine, is used.

    Raises
    ------
    ValueError
        If the selected engine is not available.
    """

    def __init__(self, engine: Literal["patsy", "formulaic"] | None = None):
        self._engine = self._get_engine(engine)
        self._using_patsy = self._engine == "patsy"
        self._spec = None
        self._missing_mask = None

    @staticmethod
    def _get_engine(
        engine: Literal["patsy", "formulaic"] | None = None,
    ) -> Literal["patsy", "formulaic"]:
        """
        Check user engine selection or get the default engine to use.

        Parameters
        ----------
        engine : {"patsy", "formulaic"} or None
            The formula engine to use. If None, the default engine, which appears
            in the attribute statsmodels.formula.formula_options.engine, is used.

        Returns
        -------
        engine : {"patsy", "formulaic"}
            The selected engine

        Raises
        ------
        ValueError
            If the selected engine is not available.
        """
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
        # Ensure selected engine is available
        msg_base = " is not available. Please install patsy."
        if _engine == "patsy" and not HAVE_PATSY:
            raise ImportError(f"patsy {msg_base}.")
        if _engine == "formulaic" and not HAVE_FORMULAIC:
            raise ImportError(f"formulaic {msg_base}.")

        return _engine

    @property
    def engine(self):
        """Get the formula engine."""
        return self._engine

    @property
    def spec(self):
        """
        Get the model specification. Only available after calling get_arrays.
        """
        return self._spec

    @property
    def factor_evaluation_error(self):
        """
        Get the engine-specific error that may occur when evaluating a factor.
        """
        if self._using_patsy:
            return patsy.PatsyError
        else:
            return formulaic.errors.FactorEvaluationError

    @property
    def formula_materializer_error(self):
        """
        Get the engine-specific error that may occur when materializing a formula.
        """
        if self._using_patsy:
            return patsy.PatsyError
        else:
            return formulaic.errors.FormulaMaterializerNotFoundError

    @property
    def missing_mask(self):
        """
        Get a mask indicating if data are missing.

        Returns
        -------
        ndarray
            A boolean array indicating if data are missing. True if missing.
        """
        return self._missing_mask

    @property
    def intercept_term(self):
        """
        Get the formula-engine-specific intercept term.

        Returns
        -------
        Term
            The intercept term.
        """
        if self._using_patsy:
            from patsy.desc import INTERCEPT

            return INTERCEPT
        else:
            from formulaic.parser.types.factor import Factor
            from formulaic.parser.types.term import Term

            return Term((Factor("1"),))

    @property
    def model_spec_type(self):
        """
        Returns the engine-specific model specification type.

        Returns
        -------
        {type[ModelSpect], type[DesignInto]}
            The model specification type of formulaic is ModelSpec.
            patsy uses DesignInfo.
        """
        if self._using_patsy:
            return patsy.design_info.DesignInfo
        else:
            return formulaic.model_spec.ModelSpec

    @staticmethod
    def _legacy_orderer(
        formula: str, data: pd.DataFrame, context: int | Mapping[str, Any]
    ) -> formulaic.Formula:
        """
        Function to order a formulaic formula so when materialized it matches patsy.

        Parameters
        ----------
        formula : str
            The string formula. If not a string, it is returned as is.
        data : DataFrame
            The data used to materialize the formula.
        context
            The context used to evaluate the formula.

        Returns
        -------
        Formula
            The ordered formula.
        """
        if isinstance(formula, (formulaic.Formula, formulaic.ModelSpec)):
            return formula
        if isinstance(context, int):
            context += 1
        mm = formulaic.model_matrix(formula, data, context=context)
        feature_flags = formulaic.parser.DefaultFormulaParser.FeatureFlags.TWOSIDED
        parser = formulaic.parser.DefaultFormulaParser(feature_flags=feature_flags)
        _formula = formulaic.Formula(formula, _parser=parser, _ordering="none")
        if isinstance(mm, formulaic.ModelMatrices):
            rhs = mm.rhs
            rhs_formula = _formula.rhs
            lhs_formula = _formula.lhs
        else:
            rhs = mm
            rhs_formula = _formula
            lhs_formula = None
        include_intercept = any(
            [(term.degree == 0 and str(term) == "1") for term in rhs_formula]
        )
        parser = formulaic.parser.DefaultFormulaParser(
            feature_flags=feature_flags, include_intercept=include_intercept
        )
        original = list(parser.get_terms(str(rhs_formula)).root)
        categorical_variables = list(rhs.model_spec.factor_contrasts.keys())

        def all_cat(term):
            return all([f in categorical_variables for f in term.factors])

        def drop_terms(term_list, terms):
            for term in terms:
                term_list.remove(term)

        intercept = [term for term in original if term.degree == 0]
        drop_terms(original, intercept)
        cats = sorted(
            [term for term in original if all_cat(term)], key=lambda term: term.degree
        )
        drop_terms(original, cats)
        conts = defaultdict(list)
        for term in original:
            cont = [
                factor for factor in term.factors if factor not in categorical_variables
            ]
            conts[tuple(sorted(cont))].append((term, term.degree))
        final_conts = []
        for key, value in conts.items():
            tmp = sorted(value, key=lambda term_degree: term_degree[1])
            final_conts.extend([value[0] for value in tmp])
        if lhs_formula is not None:
            return formulaic.Formula(
                lhs=formulaic.Formula(lhs_formula, _ordering="none"),
                rhs=formulaic.Formula(intercept + cats + final_conts, _ordering="none"),
                _ordering="none",
            )
        else:
            return formulaic.Formula(intercept + cats + final_conts, _ordering="none")

    def get_matrices(
        self,
        formula,
        data,
        eval_env=0,
        pandas=True,
        na_action=None,
        prediction=False,
    ) -> (
        np.ndarray
        | tuple[np.ndarray, np.ndarray]
        | pd.DataFrame
        | tuple[pd.DataFrame, pd.DataFrame]
    ):
        f"""
        Get the model matrices or design matrices from a formula and data.

        Parameters
        ----------
        formula : str, Formula
            The formula to use.
        data : DataFrame, dict, list, recarray
            The data to use when evaluating the formula.
        eval_env : {int, dict}
            Additional context to use when evaluating the formula.
        pandas : bool
            Return a DataFrame if true, otherwise return a numpy array.
        na_action : {NAAction, str}
            The action to take on missing values.
        prediction : bool
            True if using the formula for prediction. In this case, only the
            rhs is evaluated using data.

        Returns
        -------
        tuple[DataFrame, ...], tuple[ndarray, ...], tuple[DesignMatrix, ...]
            If pandas is True, returns one or more DataFrames. If False,
            returns a NumPy ndarray (formulaic) or a DesignMatrix (patsy).
        """
        _check_data(data)
        if isinstance(eval_env, (int, np.integer)):
            eval_env = int(eval_env) + 1
        if self._using_patsy:
            return_type = "dataframe" if pandas else "matrix"
            kwargs = {}
            if na_action:
                kwargs["NA_action"] = na_action
            if isinstance(eval_env, Mapping):
                _eval_env = patsy.eval.EvalEnvironment(
                    [{key: val} for key, val in eval_env.items()]
                )
            else:
                _eval_env = eval_env
                if isinstance(eval_env, patsy.eval.EvalEnvironment):
                    warnings.warn(EVAL_ENV_WARNING, FutureWarning)
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
            if isinstance(output, tuple):
                self._spec = output[1].design_info
            else:
                self._spec = output.design_info
            if isinstance(na_action, NAAction):
                self._missing_mask = getattr(na_action, "missing_mask", None)
            return output

        else:  # self._engine == "formulaic":
            import statsmodels.formula

            kwargs = {}
            if pandas:
                kwargs["output"] = "pandas"
            else:
                kwargs["output"] = "numpy"

            if na_action:
                kwargs["na_action"] = na_action
            if isinstance(data, (dict, Mapping)):
                # Require a DataFrame to ensure that we can handle dropped
                if all(np.isscalar(v) for v in data.values()):
                    # Handle dict of scalars
                    _data = pd.DataFrame(data, index=[0])
                else:
                    _data = pd.DataFrame(data)
            elif isinstance(data, np.rec.recarray):
                _data = pd.DataFrame.from_records(data)
            else:
                _data = data

            _eval_env = eval_env
            if HAVE_PATSY:
                from patsy.eval import EvalEnvironment

                if isinstance(eval_env, EvalEnvironment):
                    warnings.warn(EVAL_ENV_WARNING, FutureWarning)

                    ns = eval_env._namespaces
                    _eval_env = {}
                    for val in ns:
                        _eval_env.update(val)
            if not isinstance(_eval_env, (int, dict)):
                raise TypeError("context (eval_env) must be an int or a dict.")

            _ordering = statsmodels.formula.options.ordering
            if isinstance(formula, self.model_spec_type):
                _formula = formula
            if _ordering == "legacy":
                _formula = self._legacy_orderer(formula, _data, context=_eval_env)
            else:
                feature_flags = (
                    formulaic.parser.DefaultFormulaParser.FeatureFlags.TWOSIDED
                )
                parser = formulaic.parser.DefaultFormulaParser(
                    feature_flags=feature_flags
                )
                _formula = formulaic.formula.Formula(
                    formula, _ordering=_ordering, _parser=parser
                )
            if prediction:
                if hasattr(_formula, "rhs"):
                    _formula = _formula.rhs
            original_index = _data.index
            na_drop = "na_action" in kwargs and kwargs["na_action"] == "drop"
            if na_drop:
                _data = _data.copy(deep=False)
                _data.index = pd.RangeIndex(_data.shape[0])

            output = formulaic.model_matrix(
                _formula, _data, context=_eval_env, **kwargs
            )
            if na_drop:
                if isinstance(output, formulaic.ModelMatrices):
                    rhs = output.rhs
                else:
                    rhs = output
                if rhs.shape[0] == _data.shape[0]:
                    replacement_index = original_index
                    empty_mask = np.full(rhs.shape[0], False)
                    self._missing_mask = pd.Series(
                        empty_mask, index=original_index, name=None
                    )
                else:
                    replacement_index = original_index[rhs.index]
                    self._missing_mask = (
                        pd.Series(rhs.index, index=rhs.index)
                        .reindex(pd.RangeIndex(_data.shape[0]))
                        .isna()
                    )
                for key in ("lhs", "rhs", "root"):
                    if hasattr(output, key):
                        df = getattr(output, key)
                        df.index = replacement_index
            if not isinstance(output, formulaic.ModelMatrix):
                if (
                    len(output) == 2
                    and hasattr(output, "lhs")
                    and hasattr(output, "rhs")
                ):
                    self._spec = output.rhs.model_spec
                    return output.lhs, output.rhs
                else:
                    raise ValueError(
                        "The formula has produced matrices that are not currently "
                        "supported."
                    )

            self._spec = output.model_spec
            return output

    def get_linear_constraints(
        self, constraints: np.ndarray | str | Sequence[str], variable_names: list[str]
    ):
        """
        Get the linear constraints from the constraints and variable names.

        Parameters
        ----------
        constraints : ndarray, str, list[str]
            The constraints either as an array, a string, or a list of strings.
            Single strings can be comma-separated so that multiple constraints
            can be passed in this format.
        variable_names : list[str]
            The ordered list of variable names.

        Returns
        -------
        LinearConstraintValues
            The constraint matrix, constraint values, and variable names.
        """
        if self._using_patsy:
            from patsy.design_info import DesignInfo

            lc = DesignInfo(variable_names).linear_constraint(constraints)
            return LinearConstraintValues(
                constraint_matrix=lc.coefs,
                constraint_values=lc.constants,
                variable_names=lc.variable_names,
            )
        else:  # self._engine == "formulaic"
            # Handle list of constraints, which is not supported by formulaic
            if isinstance(constraints, list):
                if len(constraints) == 0:
                    raise ValueError("Constraints must be non-empty")

                if isinstance(constraints[0], str):
                    if not all(isinstance(c, str) for c in constraints):
                        raise ValueError(
                            "All constraints must be strings when passed as a list."
                        )
                    _constraints = ", ".join(str(v) for v in constraints)
                else:
                    _constraints = np.array(constraints)
            else:
                _constraints = constraints
            if isinstance(_constraints, tuple):
                _constraints = (
                    _constraints[0],
                    np.atleast_1d(np.squeeze(_constraints[1])),
                )
            lc_f = formulaic.utils.constraints.LinearConstraints.from_spec(
                _constraints, variable_names=list(variable_names)
            )
            return LinearConstraintValues(
                constraint_matrix=lc_f.constraint_matrix,
                constraint_values=np.atleast_2d(lc_f.constraint_values).T,
                variable_names=lc_f.variable_names,
            )

    def get_empty_eval_env(self):
        """
        Get an empty evaluation environment.

        Returns
        -------
        {EvalEnvironment, dict}
            A formula-engine-dependent empty evaluation environment.
        """
        if self._using_patsy:
            from patsy.eval import EvalEnvironment

            return EvalEnvironment({})
        else:
            return {}

    def remove_intercept(self, terms):
        """
        Remove intercept from Patsy terms.

        Parameters
        ----------
        terms : list[Term]
            A list of terms that might have an intercept

        Returns
        -------
        list[Term]
            The terms with the intercept removed, if present.
        """
        intercept = self.intercept_term
        if intercept in terms:
            terms.remove(intercept)
        return terms

    def has_intercept(self, spec):
        """
        Check if the model specification has an intercept term.

        Parameters
        ----------
        spec

        Returns
        -------
        bool
            True if the model specification has an intercept term, False otherwise.
        """

        return self.intercept_term in spec.terms

    def intercept_idx(self, spec):
        """
        Returns boolean array index indicating which column holds the intercept.

        Parameters
        ----------
        spec

        Returns
        -------
        ndarray
            Boolean array index indicating which column holds the intercept.
        """
        from numpy import array

        intercept = self.intercept_term
        return array([intercept == i for i in spec.terms])

    def get_na_action(self, action: str = "drop", types: Sequence[Any] = _NoDefault):
        """
        Get the NA action for the formula engine.

        Parameters
        ----------
        action
        types

        Returns
        -------
        NAAction | str
            The formula-engine-specific NA action.

        Notes
        -----
        types is ignored when using formulaic.
        """
        types = ["None", "NaN"] if types is _NoDefault else types
        if self._using_patsy:
            return NAAction(on_NA=action, NA_types=types)
        else:
            return action

    def get_spec(self, formula):
        """
        Get the model specification from a formula.

        Parameters
        ----------
        formula : str
            The formula to parse.

        Returns
        -------
        {ModelDesc, Formula}
            The engine-specific model specification.
        """
        if self._using_patsy:
            return patsy.ModelDesc.from_formula(formula)
        else:
            return formulaic.Formula(formula)

    def _ensure_spec(self, spec_or_frame):
        if not isinstance(spec_or_frame, self.model_spec_type):
            return self.get_model_spec(spec_or_frame)
        if not isinstance(spec_or_frame, self.model_spec_type):
            raise TypeError(
                "spec_or_frame must be a ModelSpec or a DesignInfo when not a "
                f"DataFrame. Expected a {self.model_spec_type.__name__}, got a "
                f"{type(spec_or_frame).__name__}."
            )
        return spec_or_frame

    def get_term_names(self, spec_or_frame):
        """
        Get the term names from a model specification or DataFrame.

        Parameters
        ----------
        spec_or_frame

        Returns
        -------

        """
        spec = self._ensure_spec(spec_or_frame)
        if self._using_patsy:
            return list(spec.term_names)
        else:
            return [str(term) for term in spec.terms]

    def get_column_names(self, spec_or_frame):
        """
        Returns a list of column names from a model specification or DataFrame.

        Parameters
        ----------
        spec_or_frame : {DataFrame, ModelSpec, DesignInfo

        Returns
        -------
        list[str]
            The column names.
        """
        spec = self._ensure_spec(spec_or_frame)
        return list(spec.column_names)

    def get_term_name_slices(self, spec_or_frame):
        """
        Get a dictionary containing the term names and their location in the formula.

        Parameters
        ----------
        spec_or_frame : {DataFrame, ModelSpec, DesignInfo}
            The DataFrame with a model specification attached or the model
            specification.

        Returns
        -------
        dict[str, slice]
            A dictionary mapping term names to location in the materialized formula.
        """
        spec = self._ensure_spec(spec_or_frame)
        if self._using_patsy:
            return dict(spec.term_name_slices)
        else:
            return spec.term_slices

    def get_model_spec(self, frame, optional=False):
        """

        Parameters
        ----------
        frame : DataFrame
            The frame to get the model specification from.
        optional : bool
            Whether to return None if the frame does not have a model specification.

        Returns
        -------
        {ModelSpec, DesignInfo}
            The engine-specific model specification
        """
        if self._using_patsy:
            if optional and not hasattr(frame, "design_info"):
                return None
            return frame.design_info
        else:
            if optional and not hasattr(frame, "model_spec"):
                return None
            return frame.model_spec

    def get_slice(self, model_spec, term):
        """

        Parameters
        ----------
        model_spec : {ModelSpec, DesignInfo}
            The model specification.
        term : Term
            The model term.

        Returns
        -------
        slice
            The slice indicating the variables connected to a specific model term.
        """
        if self._using_patsy:
            return model_spec.slice(term)
        else:
            return model_spec.get_slice(term)

    def get_term_name(self, term):
        """
        Gets the string name of a term

        Parameters
        ----------
        term : Term

        Returns
        -------
        str
            The term's name.
        """
        if self._using_patsy:
            return term.name()
        else:
            return str(term)

    def get_description(self, spec_or_frame):
        """
        Gets a string representation of the model specification.

        Parameters
        ----------
        spec_or_frame : {DataFrame, ModelSpec, DesignInfo

        Returns
        -------
        str
            The human-readable description of the model specification.,
        """
        spec = self._ensure_spec(spec_or_frame)
        if self._using_patsy:
            return spec.describe()
        else:
            return str(spec.formula)

    def get_factor_categories(self, factor, model_spec):
        """
        Get the list of categories for a factor.

        Parameters
        ----------
        factor : {EvalFactor, Factor}
            The factor to get the categories for.
        model_spec : {ModelSpec, DesignInfo}
            The model specification.

        Returns
        -------
        tuple
            The categories for the factor.
        """
        if self._using_patsy:
            return model_spec.factor_infos[factor].categories
        else:
            return tuple(model_spec.encoder_state[factor][1]["categories"])

    def get_contrast_matrix(self, term, factor, model_spec):
        """
        Get the contrast matrix for a term and factor.

        Parameters
        ----------
        term : Term
            Either a formulaic Term or a patsy Term.
        factor : EvalFactor or Factor
            Either a formulaic Factor or a patsy EvalFactor
        model_spec : engine-specific model specification
            Either a formulaic ModelSpec or a patsy DesignInfo.

        Returns
        -------
        ndarray
            The contract matrix to use for hypothesis testing.
        """
        if self._using_patsy:
            return model_spec.term_codings[term][0].contrast_matrices[factor].matrix
        else:
            cat = self.get_factor_categories(factor, model_spec)
            reduced_rank = True
            for ts in model_spec.structure:
                if ts.term == term:
                    reduced_rank = len(ts.columns) != len(cat)
                    break

            return np.asarray(
                model_spec.factor_contrasts[factor].get_coding_matrix(
                    reduced_rank=reduced_rank
                )
            )
