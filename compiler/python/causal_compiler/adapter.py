"""Statsmodels adapter for extracting causal contrast inputs."""

from dataclasses import dataclass
from numbers import Real
import re

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Logit, Poisson
from statsmodels.genmod.families import links as glm_links
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS

from .contrast import intervention_predict, treatment_effect


@dataclass(frozen=True)
class ContrastInputs:
    """Extracted arrays and parameter indices for an OLS treatment contrast.

    Attributes
    ----------
    interaction_data : ndarray
        Two-dimensional array of covariates that interact with the treatment.
    coefficients : ndarray
        Full fitted parameter vector from the OLS model.
    main_effect_index : int
        Index of the treatment main effect in ``coefficients``.
    interaction_indices : tuple of int
        Indices of the treatment-interaction terms in ``coefficients``.
    interaction_columns : tuple of str
        DataFrame column names corresponding to ``interaction_data`` columns.
    treatment_var : str
        Name of the treatment column.
    control_value : float
        Baseline (control) treatment value.
    treated_value : float
        Counterfactual (treated) treatment value.
    """

    interaction_data: np.ndarray
    coefficients: np.ndarray
    main_effect_index: int
    interaction_indices: tuple[int, ...]
    interaction_columns: tuple[str, ...]
    treatment_var: str
    control_value: float
    treated_value: float

    @property
    def treatment_delta(self) -> float:
        """Treated treatment value minus control treatment value."""
        return float(self.treated_value - self.control_value)

    @property
    def main_effect(self) -> float:
        """Fitted coefficient of the treatment main effect."""
        return float(self.coefficients[self.main_effect_index])

    @property
    def interaction_coefficients(self) -> np.ndarray:
        """Fitted coefficients for the treatment-interaction terms."""
        return self.coefficients[list(self.interaction_indices)]


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)


def extract_ols_contrast(
    results,
    treatment_var: str,
    treatment_values: tuple[Real, Real] | list[Real],
    data: pd.DataFrame,
) -> ContrastInputs:
    """Extract contrast inputs from a fitted formula-based OLS result.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted ``statsmodels`` formula OLS results object.
    treatment_var : str
        Name of the numeric treatment column.
    treatment_values : tuple of float or list of float
        Pair ``(control_value, treated_value)`` of distinct finite numbers.
    data : pandas.DataFrame
        Scoring dataset containing the treatment and interaction columns.

    Returns
    -------
    ContrastInputs
        Extracted arrays, parameter indices, and treatment delta.

    Raises
    ------
    TypeError
        If inputs have unsupported types or non-numeric columns.
    ValueError
        If the model is not formula-based OLS, lacks required terms, or has
        unsupported transformed/higher-order treatment interactions.
    """
    model = getattr(results, "model", None)
    if not isinstance(model, OLS):
        raise TypeError("results must be fitted OLS results")
    if not hasattr(results, "params"):
        raise ValueError("results must contain fitted parameters")
    if not hasattr(model, "formula"):
        raise ValueError("results must come from a formula-based OLS model")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one row")
    if not isinstance(treatment_var, str) or not treatment_var:
        raise TypeError("treatment_var must be a non-empty column name")
    if treatment_var not in data.columns:
        raise ValueError(f"treatment column {treatment_var!r} is missing from data")
    if not pd.api.types.is_numeric_dtype(data[treatment_var]):
        raise TypeError("treatment column must contain numeric values")
    if (
        not isinstance(treatment_values, (tuple, list))
        or len(treatment_values) != 2
        or not all(_is_finite_real(v) for v in treatment_values)
        or treatment_values[0] == treatment_values[1]
    ):
        raise ValueError(
            "treatment_values must be a pair of distinct finite numeric values"
        )

    names = list(model.exog_names)
    if treatment_var not in names:
        raise ValueError(f"model must include a main effect for {treatment_var!r}")

    interaction_columns: list[str] = []
    interaction_indices: list[int] = []
    pattern = re.compile(rf"\b{re.escape(treatment_var)}\b")

    for index, name in enumerate(names):
        if name == treatment_var:
            continue
        factors = name.split(":")
        if not any(pattern.search(factor) for factor in factors):
            continue
        if treatment_var not in factors:
            raise ValueError("transformed treatment terms are not supported")
        if len(factors) != 2 or factors.count(treatment_var) != 1:
            raise ValueError(
                "only two-factor interactions with one treatment factor are supported"
            )
        col = factors[1] if factors[0] == treatment_var else factors[0]
        if col not in data.columns:
            raise ValueError("interaction factors must be raw numeric data columns")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise TypeError("interaction columns must contain numeric values")
        interaction_columns.append(col)
        interaction_indices.append(index)

    if not interaction_columns:
        raise ValueError("the model must include at least one treatment interaction")

    coefficients = np.asarray(results.params, dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size != len(names):
        raise ValueError("fitted parameters do not match model design columns")
    if not np.isfinite(coefficients).all():
        raise ValueError("fitted coefficients must all be finite")

    interaction_data = data[interaction_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(interaction_data).all():
        raise ValueError("interaction data must contain only finite values")

    control_value, treated_value = treatment_values
    return ContrastInputs(
        interaction_data=interaction_data,
        coefficients=coefficients,
        main_effect_index=names.index(treatment_var),
        interaction_indices=tuple(interaction_indices),
        interaction_columns=tuple(interaction_columns),
        treatment_var=treatment_var,
        control_value=float(control_value),
        treated_value=float(treated_value),
    )


def evaluate_ols_contrast(
    results,
    treatment_var: str,
    treatment_values: tuple[Real, Real] | list[Real],
    data: pd.DataFrame,
) -> pd.Series:
    """Compute row-level OLS treatment contrasts from a fitted model.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted ``statsmodels`` formula OLS results object.
    treatment_var : str
        Name of the numeric treatment column.
    treatment_values : tuple of float or list of float
        Pair ``(control_value, treated_value)`` of distinct finite numbers.
    data : pandas.DataFrame
        Scoring dataset containing the treatment and interaction columns.

    Returns
    -------
    pandas.Series
        Predicted treated-minus-control contrast indexed like ``data``.
    """
    inputs = extract_ols_contrast(results, treatment_var, treatment_values, data)
    effects = treatment_effect(
        inputs.interaction_data,
        main_effect=inputs.main_effect,
        interaction_coefficients=inputs.interaction_coefficients,
        treatment_delta=inputs.treatment_delta,
    )
    return pd.Series(effects, index=data.index, name="treatment_effect")


@dataclass(frozen=True)
class InterventionInputs:
    """Extracted arrays and parameter indices for K-dose intervention prediction.

    Attributes
    ----------
    covariate_data : ndarray
        Two-dimensional ``(N, D)`` matrix of baseline covariates.
    coefficients : ndarray
        Full fitted parameter vector ``(P,)``.
    base_indices : tuple of int
        Indices in ``coefficients`` for each covariate's main effect (``-1`` if absent).
    interaction_indices : tuple of int
        Indices in ``coefficients`` for each covariate's ``T:x`` term (``-1`` if absent).
    covariate_columns : tuple of str
        DataFrame column names corresponding to ``covariate_data`` columns.
    intercept_index : int
        Index of ``Intercept`` in ``coefficients`` (``-1`` if absent).
    treatment_index : int
        Index of ``treatment_var`` in ``coefficients``.
    treatment_var : str
        Name of the treatment column.
    doses : tuple of float
        ``K`` treatment values to evaluate.
    link : str
        Link name (``"identity"``, ``"logit"``, or ``"log"``).
    """

    covariate_data: np.ndarray
    coefficients: np.ndarray
    base_indices: tuple[int, ...]
    interaction_indices: tuple[int, ...]
    covariate_columns: tuple[str, ...]
    intercept_index: int
    treatment_index: int
    treatment_var: str
    doses: tuple[float, ...]
    link: str


def _detect_link(model) -> str:
    if isinstance(model, OLS):
        return "identity"
    if isinstance(model, Logit):
        return "logit"
    if isinstance(model, Poisson):
        return "log"
    if isinstance(model, GLM):
        link = model.family.link
        if isinstance(link, glm_links.Identity):
            return "identity"
        if isinstance(link, glm_links.Logit):
            return "logit"
        if isinstance(link, glm_links.Log):
            return "log"
        raise TypeError(f"unsupported GLM link: {type(link).__name__}")
    raise TypeError(
        f"unsupported statsmodels model {type(model).__name__}; "
        "expected OLS, Logit, Poisson, or GLM"
    )


def extract_intervention_inputs(
    results,
    treatment_var: str,
    doses: tuple[Real, ...] | list[Real],
    data: pd.DataFrame | np.ndarray,
) -> InterventionInputs:
    """Extract K-dose prediction inputs from a fitted OLS, Logit, Poisson, or GLM.

    ``data`` may be a ``pandas.DataFrame`` or a pre-extracted 2-D ``ndarray``
    whose columns match the model's covariates in ``exog_names`` order.
    """
    model = getattr(results, "model", None)
    link = _detect_link(model)
    if not hasattr(results, "params"):
        raise ValueError("results must contain fitted parameters")
    if not hasattr(model, "formula"):
        raise ValueError("results must come from a formula-based model")
    is_df = isinstance(data, pd.DataFrame)
    if is_df:
        if data.empty:
            raise ValueError("data must be a non-empty pandas DataFrame")
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("ndarray data must be a non-empty 2-D array")
    if not isinstance(treatment_var, str) or not treatment_var:
        raise TypeError("treatment_var must be a non-empty column name")
    if is_df and treatment_var not in data.columns:
        raise ValueError(f"treatment column {treatment_var!r} is missing from data")
    if (
        not isinstance(doses, (tuple, list))
        or len(doses) == 0
        or not all(_is_finite_real(d) for d in doses)
    ):
        raise ValueError("doses must be a non-empty sequence of finite numbers")

    names = list(model.exog_names)
    if treatment_var not in names:
        raise ValueError(f"model must include a main effect for {treatment_var!r}")

    intercept_index = names.index("Intercept") if "Intercept" in names else -1
    treatment_index = names.index(treatment_var)
    pattern = re.compile(rf"\b{re.escape(treatment_var)}\b")

    base_map: dict[str, int] = {}
    int_map: dict[str, int] = {}
    cov_order: list[str] = []

    def _register_cov(col: str) -> None:
        if col not in base_map and col not in int_map:
            if is_df:
                if col not in data.columns:
                    raise ValueError(f"covariate column {col!r} is missing from data")
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise TypeError("covariate columns must contain numeric values")
            cov_order.append(col)

    for index, name in enumerate(names):
        if name in ("Intercept", treatment_var):
            continue
        factors = name.split(":")
        if any(pattern.search(factor) for factor in factors):
            if treatment_var not in factors:
                raise ValueError("transformed treatment terms are not supported")
            if len(factors) != 2 or factors.count(treatment_var) != 1:
                raise ValueError(
                    "only two-factor interactions with one treatment factor are supported"
                )
            col = factors[1] if factors[0] == treatment_var else factors[0]
            _register_cov(col)
            int_map[col] = index
        else:
            if len(factors) != 1:
                raise ValueError(f"unsupported multi-factor covariate term {name!r}")
            _register_cov(name)
            base_map[name] = index

    if not cov_order:
        raise ValueError("the model must include at least one covariate")

    coefficients = np.asarray(results.params, dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size != len(names):
        raise ValueError("fitted parameters do not match model design columns")
    if not np.isfinite(coefficients).all():
        raise ValueError("fitted coefficients must all be finite")

    if is_df:
        covariate_data = data[cov_order].to_numpy(dtype=np.float64)
    else:
        covariate_data = np.asarray(data, dtype=np.float64)
        if covariate_data.shape[1] != len(cov_order):
            raise ValueError(
                f"ndarray data has {covariate_data.shape[1]} columns, expected {len(cov_order)}"
            )
    if not np.isfinite(covariate_data).all():
        raise ValueError("covariate data must contain only finite values")

    return InterventionInputs(
        covariate_data=covariate_data,
        coefficients=coefficients,
        base_indices=tuple(base_map.get(c, -1) for c in cov_order),
        interaction_indices=tuple(int_map.get(c, -1) for c in cov_order),
        covariate_columns=tuple(cov_order),
        intercept_index=intercept_index,
        treatment_index=treatment_index,
        treatment_var=treatment_var,
        doses=tuple(float(d) for d in doses),
        link=link,
    )


def evaluate_intervention_predict(
    results,
    treatment_var: str,
    doses: tuple[Real, ...] | list[Real],
    data: pd.DataFrame | np.ndarray,
    reduction: str = "none",
) -> np.ndarray:
    """Evaluate K treatment doses for a fitted OLS, Logit, Poisson, or GLM."""
    inputs = extract_intervention_inputs(results, treatment_var, doses, data)
    return intervention_predict(
        inputs.covariate_data,
        inputs.coefficients,
        base_indices=inputs.base_indices,
        interaction_indices=inputs.interaction_indices,
        intercept_index=inputs.intercept_index,
        treatment_index=inputs.treatment_index,
        doses=inputs.doses,
        link=inputs.link,
        reduction=reduction,
    )
