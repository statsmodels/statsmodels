import numpy as np
from patsy import NAAction

from statsmodels.formula._manager import FormulaManager
import statsmodels.tools.data as data_util

# if users want to pass in a different formula framework, they can
# add their handler here. how to do it interactively?

# this is a mutable object, so editing it should show up in the below
formula_handler = {}

# TODO: patsy migration
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


def handle_formula_data(Y, X, formula, depth=0, missing='drop'):
    """
    Returns endog, exog, and the model specification from arrays and formula.

    Parameters
    ----------
    Y : array_like
        Either endog (the LHS) of a model specification or all of the data.
        Y must define __getitem__ for now.
    X : array_like
        Either exog or None. If all the data for the formula is provided in
        Y then you must explicitly set X to None.
    formula : str or patsy.model_desc
        You can pass a handler by import formula_handler and adding a
        key-value pair where the key is the formula object class and
        the value is a function that returns endog, exog, formula object.

    Returns
    -------
    endog : array_like
        Should preserve the input type of Y,X.
    exog : array_like
        Should preserve the input type of Y,X. Could be None.
    """
    # half ass attempt to handle other formula objects
    if isinstance(formula, tuple(formula_handler.keys())):
        return formula_handler[type(formula)]

    na_action = NAAction(on_NA=missing)
    mgr = FormulaManager()
    if X is not None:
        result = mgr.get_arrays(formula, (Y, X), eval_env=depth, pandas=True, na_action=na_action, attach_spec=True)
    else:
        result = mgr.get_arrays(formula, Y, eval_env=depth, pandas=True, na_action=na_action, attach_spec=True)

    # if missing == 'raise' there's not missing_mask
    missing_mask = getattr(na_action, 'missing_mask', None)
    if not np.any(missing_mask):
        missing_mask = None
    if len(result) > 1:  # have RHS design
        design_info = mgr.spec  # detach it from DataFrame
    else:
        design_info = None
    # NOTE: is there ever a case where we'd need LHS design_info?
    return result, missing_mask, design_info




def make_hypotheses_matrices(model_results, test_formula):
    """
    """
    from statsmodels.formula._manager import FormulaManager
    mgr = FormulaManager()

    exog_names = model_results.model.exog_names
    lc = mgr.get_linear_constraints(test_formula, exog_names)
    return lc
