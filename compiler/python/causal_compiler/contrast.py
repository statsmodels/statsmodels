"""Reference implementations for causal contrast calculations."""

import numpy as np


def treatment_effect(
    interaction_data: np.ndarray,
    main_effect: float,
    interaction_coefficients: np.ndarray,
    treatment_delta: float,
) -> np.ndarray:
    """Compute a row-level OLS treatment contrast from interaction terms.

    The operation is:

    ``treatment_delta * (main_effect + interaction_data @ interaction_coefficients)``

    Parameters
    ----------
    interaction_data : ndarray
        Matrix with one observation per row and one treatment-interaction
        feature per column.
    main_effect : float
        Fitted coefficient of the treatment main effect.
    interaction_coefficients : ndarray
        Fitted coefficients corresponding to the columns of
        ``interaction_data``.
    treatment_delta : float
        Treated treatment value minus control treatment value.

    Returns
    -------
    ndarray
        Treatment contrast for each observation.

    Raises
    ------
    ValueError
        If the input array dimensions or feature counts do not match.
    """
    interaction_data = np.asarray(interaction_data, dtype=np.float64)
    interaction_coefficients = np.asarray(
        interaction_coefficients, dtype=np.float64
    )
    if interaction_data.ndim != 2:
        raise ValueError("interaction_data must be a two-dimensional array")
    if interaction_coefficients.ndim != 1:
        raise ValueError(
            "interaction_coefficients must be a one-dimensional array"
        )
    if interaction_data.shape[1] != interaction_coefficients.size:
        raise ValueError(
            "interaction_data columns must match interaction_coefficients"
        )

    return treatment_delta * (
        main_effect + interaction_data @ interaction_coefficients
    )


def intervention_predict(
    covariate_data: np.ndarray,
    coefficients: np.ndarray,
    base_indices: tuple[int, ...] | list[int],
    interaction_indices: tuple[int, ...] | list[int],
    intercept_index: int,
    treatment_index: int,
    doses: tuple[float, ...] | list[float],
    link: str = "logit",
    reduction: str = "none",
) -> np.ndarray:
    """Evaluate K treatment doses in one pass by hoisting ``X @ [beta_x, beta_int]``.

    Parameters
    ----------
    covariate_data : ndarray
        Two-dimensional ``(N, D)`` array of baseline covariates.
    coefficients : ndarray
        One-dimensional ``(P,)`` fitted model parameter vector.
    base_indices : sequence of int
        Indices in ``coefficients`` for each covariate's main effect
        (``-1`` if absent).
    interaction_indices : sequence of int
        Indices in ``coefficients`` for each covariate's ``T:x`` interaction
        (``-1`` if absent).
    intercept_index : int
        Index of the intercept in ``coefficients`` (``-1`` if absent).
    treatment_index : int
        Index of the treatment main effect in ``coefficients``.
    doses : sequence of float
        ``K`` treatment values to evaluate.
    link : {"identity", "logit", "log"}, optional
        Model link function.
    reduction : {"none", "ate", "argmax"}, optional
        Output mode: ``"none"`` returns ``(N, K)``, ``"ate"`` returns ``(K,)``
        column means, and ``"argmax"`` returns ``(N,)`` best-dose indices.

    Returns
    -------
    ndarray
        Predicted outcomes, dose ATEs, or per-row optimal dose indices.
    """
    x = np.asarray(covariate_data, dtype=np.float64)
    coeffs = np.asarray(coefficients, dtype=np.float64)
    dose_vec = np.asarray(doses, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("covariate_data must be a two-dimensional array")
    if coeffs.ndim != 1:
        raise ValueError("coefficients must be a one-dimensional array")
    if dose_vec.ndim != 1 or dose_vec.size == 0:
        raise ValueError("doses must be a non-empty one-dimensional sequence")
    if len(base_indices) != x.shape[1] or len(interaction_indices) != x.shape[1]:
        raise ValueError(
            "base_indices and interaction_indices must match covariate_data columns"
        )

    beta_x = np.array(
        [coeffs[idx] if idx >= 0 else 0.0 for idx in base_indices], dtype=np.float64
    )
    beta_int = np.array(
        [coeffs[idx] if idx >= 0 else 0.0 for idx in interaction_indices],
        dtype=np.float64,
    )
    weights = np.column_stack([beta_x, beta_int])
    shared = x @ weights

    beta_0 = coeffs[intercept_index] if intercept_index >= 0 else 0.0
    beta_t = coeffs[treatment_index]

    eta = (shared[:, [0]] + beta_0) + (shared[:, [1]] + beta_t) * dose_vec[None, :]

    if link == "identity":
        preds = eta
    elif link == "logit":
        preds = 1.0 / (1.0 + np.exp(-eta))
    elif link == "log":
        preds = np.exp(eta)
    else:
        raise ValueError(f"unsupported link {link!r}")

    if reduction == "none":
        return preds
    if reduction == "ate":
        return preds.mean(axis=0)
    if reduction == "argmax":
        return np.argmax(preds, axis=1).astype(np.int64)
    raise ValueError(f"unsupported reduction {reduction!r}")
