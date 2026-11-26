import warnings

import numpy as np

from statsmodels.formula._manager import FormulaManager
from statsmodels.tools.data import _to_pandas

__all__ = ["advance_eval_env", "formula_handler", "handle_formula_data"]

# Deprecated: maps the class of a formula object to a function that
# handle_formula_data calls instead of the formula engine
formula_handler = {}

FORMULA_HANDLER_WARNING = """\
formula_handler is deprecated and will be removed after 0.17. Pass a formula
string, or a formula object of the formula engine in use (patsy or formulaic).
"""


def _get_formula_handler(formula):
    handler = formula_handler.get(type(formula))
    if handler is None:
        for cls, func in formula_handler.items():
            if isinstance(formula, cls):
                return func
    return handler


def handle_formula_data(Y, X, formula, depth=0, missing="drop"):
    """
    Returns endog, exog, and the model specification from arrays and formula.

    Parameters
    ----------
    Y : DataFrame
        The data used to evaluate the formula. A dict or a recarray is
        accepted but deprecated.
    X : None
        Must be None: all of the data are passed in ``Y``.

        .. versionchanged:: 0.15.1

           Other values raise a ValueError. They never worked: the formula
           engine could not look variables up in the pair ``(Y, X)``.
    formula : str, ModelDesc or Formula
        The formula, as a string or as a formula object of the formula engine
        in use (a patsy ``ModelDesc`` or a formulaic ``Formula``).

        .. deprecated:: 0.15.1

           A formula object of another type can be handled by a function
           registered in ``formula_handler``, a dict that maps the class of
           the formula object to the function. It is called with
           ``(Y, X, formula, depth=depth, missing=missing)`` and must return
           what this function returns. The registry is deprecated and will
           be removed after 0.17. Before 0.15.1, the registered function was
           returned instead of called.
    depth : int or dict, optional
        The number of stack frames to go up from this function to the
        namespace in which variables that are not in ``Y`` are looked up (1
        is the caller's namespace), or a dict that maps names to values.
    missing : {"drop", "raise"}, optional
        The action to take on missing values.

    Returns
    -------
    result : DataFrame or tuple of DataFrame
        The design matrices: ``(endog, exog)`` for a two-sided formula, or
        the design matrix of a one-sided formula.
    missing_mask : ndarray, Series, or None
        Boolean mask that is True for the observations dropped because of
        missing values, or None if none were dropped.
    model_spec : ModelSpec or DesignInfo
        The model specification of the right-hand side.

        .. versionchanged:: 0.15.1

           It is also returned for data with a single row, which returned
           None.
    """
    handler = _get_formula_handler(formula)
    if handler is not None:
        warnings.warn(FORMULA_HANDLER_WARNING, FutureWarning, stacklevel=2)
        return handler(Y, X, formula, depth=depth, missing=missing)

    if X is not None:
        raise ValueError(
            "X must be None. Pass all of the variables used in the formula in Y."
        )

    na_action = FormulaManager().get_na_action(action=missing)
    mgr = FormulaManager()
    # Objects that support the dataframe API should be converted to a
    # dataframe to avoid problems with patsy. (This also works for
    # dataframes themselves.)
    # _to_pandas converts Polars DataFrames/Series to pandas
    Y = _to_pandas(Y)
    result = mgr.get_matrices(
        formula,
        Y,
        eval_env=depth,
        pandas=True,
        na_action=na_action,
    )

    missing_mask = mgr.missing_mask
    if not np.any(missing_mask):
        missing_mask = None
    # The spec of the right-hand side, which is the only matrix of a
    # one-sided formula
    model_spec = mgr.spec
    return result, missing_mask, model_spec


def make_hypotheses_matrices(model_results, test_formula):
    """
    Get the linear constraint matrices for a hypothesis test formula.

    Parameters
    ----------
    model_results : Results
        A results instance with an attached model that defines exog_names.
    test_formula : str
        The hypothesis test formula, e.g., "x1 = x2 = 0".

    Returns
    -------
    LinearConstraintValues
        The constraint matrix, constraint values, and variable names.
    """
    from statsmodels.formula._manager import FormulaManager

    mgr = FormulaManager()

    exog_names = model_results.model.exog_names
    lc = mgr.get_linear_constraints(test_formula, exog_names)
    return lc


def advance_eval_env(kwargs):
    """
    Adjust ``eval_env`` for a ``from_formula`` that calls its parent's.

    Calling the parent class's ``from_formula`` adds a stack frame, so
    ``eval_env`` is moved one frame up to keep pointing at the same
    namespace: None (the namespace that calls ``from_formula``) becomes 2, an
    integer k becomes k + 1, and -1 becomes an empty namespace. A dict is
    kept. Adjustments are made in place.

    Parameters
    ----------
    kwargs : dict
        The dictionary of keyword arguments passed to `from_formula`.

    Notes
    -----
    .. versionchanged:: 0.15.1

       Integers other than -1 are moved up too. Before, they were kept, so
       an integer referred to a namespace one frame lower than in the
       parent's ``from_formula``.
    """

    eval_env = kwargs.get("eval_env", None)
    if eval_env is None:
        kwargs["eval_env"] = 2
    elif eval_env == -1:
        kwargs["eval_env"] = FormulaManager().get_empty_eval_env()
    elif isinstance(eval_env, (int, np.integer)):
        kwargs["eval_env"] = int(eval_env) + 1


def get_matrices_eval_env(eval_env):
    """
    Translate the ``eval_env`` of ``from_formula`` for ``get_matrices``.

    For a formula that ``from_formula`` evaluates itself with
    ``FormulaManager.get_matrices``, for example a second formula of the
    model, so that it uses the same namespace as the model's formula.

    Parameters
    ----------
    eval_env : int, dict or None
        The ``eval_env`` passed to ``from_formula``.

    Returns
    -------
    int or dict
        The ``eval_env`` to pass to ``get_matrices`` when it is called in the
        body of ``from_formula``: 1 (the namespace that calls
        ``from_formula``) for None, an empty namespace for -1, and the value
        itself otherwise.
    """
    if eval_env is None:
        return 1
    if eval_env == -1:
        return FormulaManager().get_empty_eval_env()
    return eval_env
