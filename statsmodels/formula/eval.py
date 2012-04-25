"""
Takes a test syntax tree from parser.parse and evaluates it. This file also
includes some helper function for making linear matrix hypotheses.
"""
from numpy import zeros_like, asarray
from charlton.util import to_unique_tuple
import charlton.builtins
from charlton.eval import EvalEnvironment
from charlton.parse import ParseNode
from parser import parse

def _flatten_lists(lists):
    """
    Flattens pure lists.

    Examples
    --------
    >>> v = [[[1, 2], 3], 4]
    >>> _flatten_lists(v)
    [1, 2, 3, 4]
    >>> v = [1, [2,3], 4]
    >>> _flatten_lists(v)
    [1, 2, 3, 4]
    >>>: v = [1, [[2,3],4], 5]
    >>> _flatten_lists(v)
    [1, 2, 3, 4, 5]
    """
    new_list = []
    for item in lists:
        if isinstance(item, list):
            new_list.extend(_flatten_lists(item))
        else:
            new_list.append(item)
    return new_list

def _unnest(lists):
    """
    Used to unnest compound hypotheses. Used in Evaluator.eval only.
    """
    new_list = []
    # the inner-most list should have a tuple as item 1
    while lists and isinstance(lists[1], list):
        new_list.insert(0, lists.pop(1))
        lists = lists[0]
    # grab the last one
    new_list.insert(0, lists)
    return new_list

def _is_a_float(v):
    try:
        float(v)
    except ValueError:
        return False
    else:
        return True

def _maybe_simplify_rhs(rhs):
    if not rhs:
        rhs = (None, 0.0)
    else:
        add_scalars = lambda x, y: (None, x[1]+y[1])
        if isinstance(rhs, list):
            rhs = reduce(add_scalars, rhs)
    return rhs

def _add_like_terms(x, y=None):
    if y is None:
        return tuple(x)
    return (x[0], x[1] + y[1])

def _maybe_simplify_lhs(lhs):
    # need this for deterministic tests
    # might be able to rewrite tests and just get rid of sorting
    if isinstance(lhs, list):
        new_lhs = []
        #TODO: is there a lighter way to do this?
        unique_terms = sorted(list(set(i[0] for i in lhs)))
        terms = asarray([i[0] for i in lhs])
        lhs_arr = asarray(lhs, dtype='O')
        for term in unique_terms:
            new_lhs.append(_add_like_terms(*tuple(lhs_arr[term==terms])))
        lhs = new_lhs
    return lhs

def _maybe_rearrange_terms(exprs):
    """
    Takes an expression and puts all terms with parameters on the LHS
    and all numerical terms on the RHS then simplify RHS.
    """
    lhs = []
    rhs = []
    #from IPython.core.debugger import Pdb; Pdb().set_trace()

    # do LHS
    if isinstance(exprs[0], list):
        for term in exprs[0]:
            if term[0] is None and Evaluator._is_a(float, term[1]):
                rhs += [(term[0], term[1] * -1)]
            else:
                lhs += [term]
    elif exprs[0][0] is None:
        # then we know it's a scalar only on LHS, flip the expression
        # and start over, careful for infinite recursion if two scalars only
        # should be caught earlier though...
        return _maybe_rearrange_terms(exprs[::-1])
    else: # single term on LHS not a scalar
        lhs = [exprs[0]]

    # do RHS
    if isinstance(exprs[1], list):
        for term in exprs[1]:
            if term[0] is None and Evaluator._is_a(float, term[1]):
                rhs += [term]
            else:
                lhs += [(term[0], term[1] * -1)]
    elif exprs[1][0] is not None: # it's a term
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        lhs += [(exprs[1][0], exprs[1][1] * -1)]
    else:
        rhs += [exprs[1]]

    #from IPython.core.debugger import Pdb; Pdb().set_trace()

    rhs = _maybe_simplify_rhs(rhs)
    # make sure you give this a flattened list
    # things like 'x1 + 2*x2 + x3' get nested
    # likely treates the symptom and not the problem though
    lhs = _maybe_simplify_lhs(_flatten_lists(lhs))

    #TODO: maybe remove?
    if len(lhs) == 1: # to be consistent with other single term LHS
        lhs = lhs[0]

    return [lhs, rhs]

def _eval_binary_ampersand(evaluator, tree):
    exprs = [evaluator.eval(arg, dirty=True) for arg in tree.args]
    return exprs

def _eval_any_equals(evaluator, tree):
    exprs = [evaluator.eval(arg) for arg in tree.args]
    if len(exprs) == 1:
        # Formula was like: "= foo"
        # We pretend that instead it was like: "foo = 0"
        exprs.append((None,0))
    assert len(exprs) == 2

    exprs = _maybe_rearrange_terms(exprs)

    return exprs

def _eval_binary_plus(evaluator, tree):
    exprs = [evaluator.eval(arg) for arg in tree.args]
    assert len(exprs) == 2
    return exprs

def _eval_binary_minus(evaluator, tree):
    left_expr = evaluator.eval(tree.args[0])
    right_expr = evaluator.eval(tree.args[1])
    if isinstance(right_expr, tuple):
        right_expr = right_expr[0], right_expr[1] * -1
    return [left_expr, right_expr]

def _eval_binary_prod(evaluator, tree):
    # assuming here that we only get a linear predictor not a nonlinear
    # hypothesis
    which_float = map(_is_a_float, tree.args)
    try:
        assert any(which_float)
    except AssertionError, err:
        raise ValueError("Non-linear hypotheses not handled: %s" %
                         tree.origin.code)
    # what if we get something like 2*(x1 + x3) ? we shouldn't but ...
    return (tree.args[which_float.index(False)],
            float(tree.args[which_float.index(True)]))

def _eval_binary_div(evaluator, tree):
    which_float = map(_is_a_float, tree.args)
    try:
        assert any(which_float)
    except AssertionError, err:
        raise ValueError("Non-linear hypotheses not handled: %s" %
                         tree.origin.code)
    return (tree.args[which_float.index(False)],
            1/float(tree.args[which_float.index(True)]))

#def _eval_binary_power(evaluator, tree):
#    left_expr = evaluator.eval(tree.args[0])
#    _check_interactable(left_expr)
#    power = -1
#    try:
#        power = int(tree.args[1])
#    except (ValueError, TypeError):
#        pass
#    if power < 1:
#        raise CharltonError("'**' requires a positive integer", tree.args[1])
#    all_terms = left_expr.terms
#    big_expr = left_expr
#    # Small optimization: (a + b)**100 is just the same as (a + b)**2.
#    power = min(len(left_expr.terms), power)
#    for i in xrange(1, power):
#        big_expr = _interaction(left_expr, big_expr)
#        all_terms = all_terms + big_expr.terms
#    return IntermediateExpr(False, None, False, all_terms)

def _eval_unary_plus(evaluator, tree):
    return evaluator.eval(tree.args[0]), 1

def _eval_unary_minus(evaluator, tree):
    expr = evaluator.eval(tree.args[0])
    expr = expr[0], expr[1] * -1
    return expr

class Evaluator(object):
    def __init__(self, factor_eval_env):
        self._factor_eval_env = factor_eval_env
        self._evaluators = {}
        self.add_op("=", 2, _eval_any_equals)
        self.add_op("=", 1, _eval_any_equals)

        self.add_op("&", 2, _eval_binary_ampersand)

        self.add_op("+", 2, _eval_binary_plus)
        self.add_op("-", 2, _eval_binary_minus)
        self.add_op("*", 2, _eval_binary_prod)
        self.add_op("/", 2, _eval_binary_div)
        #self.add_op("**", 2, _eval_binary_power)

        self.add_op("+", 1, _eval_unary_plus)
        self.add_op("-", 1, _eval_unary_minus)

        self.stash = {}

    # This should not be considered a public API yet (to use for actually
    # adding new operator semantics) because I wrote in some of the relevant
    # code sort of speculatively, but it isn't actually tested.
    def add_op(self, op, arity, evaluator):
        self._evaluators[op, arity] = evaluator

    @classmethod
    def _is_a(cls, f, v):
        try:
            f(v)
        except ValueError:
            return False
        else:
            return True

    def eval(self, tree, dirty=False):
        """
        The dirty keyword is used to indicate that eval is being called
        in an multiple hypothesis context. If dirty is True, we are not on
        the 'main' call to eval. In this case, eval is being called by
        _eval_binary_ampersand.
        """
        result = None

        if isinstance(tree, str):
            if self._is_a(int, tree) or self._is_a(float, tree):
                # this is a RHS variable
                result = None, float(tree)
            else:
                # Guess it's a parameter with unitary coefficient
                result = tree, 1
        elif isinstance(tree, tuple): # got a param, coeff pair
            result = tree
        else:
            assert isinstance(tree, ParseNode)
            # sometimes args for unary operators are just ParseNodes
            if not isinstance(tree.args, list):
                tree.args = [tree.args]
            key = (tree.op.token, len(tree.args))

            if key not in self._evaluators:
                raise CharltonError("I don't know how to evaluate "
                                "this '%s' operator" % (tree.op.token,),
                                tree.op)
            result = self._evaluators[key](self, tree)
            if tree.op.token == '&' and not dirty:
                # if more than 1 &, got a nested result
                result = _unnest(result)
                result = dict(zip(range(len(result)), result))
        return result

## Where to put the below?

def _get_R(lhs, names):
    arr = zeros_like(names, dtype=float)
    if not isinstance(lhs, list):
        lhs = [lhs]
    for term in lhs:
        arr[names.index(term[0])] = term[1]
    return arr

def make_hypotheses_matrices(model_results, test_formula, depth=0):
    """
    Makes matrices for Wald tests of linear hypotheses about the parameters.

    Parameters
    ----------
    model_results : Results instance
        A model results instance.
    test_formula : str
        A string for a test formula.

    Notes
    -----
    .. todo:: Extended notes on the testing syntax.
    """
    tree = parse(test_formula)
    #we shouldn't need any of the builtins for testing
    #eval_env.add_outer_namespace(charlton.builtins.builtins)
    #do we even need an eval environment?
    test_desc = Evaluator({}).eval(tree)
    exog_names = model_results.model.exog_names
    if isinstance(test_desc, list):
        lhs, rhs = test_desc
        Q = rhs[1]
        R = _get_R(lhs, exog_names)
    elif isinstance(test_desc, dict):
        n_tests = len(test_desc)
        R, Q = [], []
        for i in range(n_tests):
            lhs, rhs = test_desc[i]
            Q.append(rhs[1])
            R.append(_get_R(lhs, exog_names))

    return asarray(R), asarray(Q)

def _do_eval_test(actual, expected):
    length = len(actual)
    if isinstance(actual, dict):
        for i in range(length):
            _do_eval_test(actual[i], expected[i])
            #assert actual[i] == expected[i]
    else:
        assert actual == expected

def test_eval():
    for code, expected in _test_eval.iteritems():
        tree = parse(code)
        actual = Evaluator({}).eval(tree)
        _do_eval_test(actual, expected)

#TODO: avoid this list / no-list difference for single-term hypotheses
_test_eval = {
        # some basic tests

        # results are parameter, it's coefficient plus the RHS term
        'x2 = 0' : [('x2', 1), (None, 0)],
        '0 = x2' : [('x2', 1), (None, 0)],
        'x1 + x2 = 0' : [[('x1', 1), ('x2', 1)], (None, 0)],
        'x1 = x2' : [[('x1', 1), ('x2', -1)], (None, 0)],
        '0 = x1 + x2' : [[('x1', 1), ('x2', 1)], (None, 0)],
        'x1 - x2 = 0' : [[('x1', 1), ('x2', -1)], (None, 0)],
        '-x2 = 1' : [('x2', -1), (None, 1)],
        'x1 + x2 = x3' : [[('x1', 1), ('x2', 1), ('x3', -1)], (None, 0)],
        'x1 + 2*x2 +x3 = 0' : [[('x1', 1), ('x2', 2), ('x3', 1)], (None, 0)],
        # takes advantage of sympy
        '-(x1 + x2) = 1' : [[('x1', -1), ('x2', -1)], (None, 1)],
        # needs rhs -> lhs
        'x1 + x2 = x3 + x4' : [[('x1', 1), ('x2', 1),
                                ('x3', -1), ('x4', -1)], (None, 0)],
        'x1 + 2*x2 = x3 - 3*x4' : [[('x1', 1), ('x2', 2),
                                ('x3', -1), ('x4', 3)], (None, 0)],
        'x1 + 5 = 3' : [('x1', 1), (None, -2)],
        'x1 + 3 - 2' : [('x1', 1), (None, -1)],
        #TODO: check that sympy can simply LHS and RHS
        # multiple hypotheses
        'x1 & x2 & x3 & x4 & x5' : {
                0 : [('x1', 1), (None, 0)],
                1 : [('x2', 1), (None, 0)],
                2 : [('x3', 1), (None, 0)],
                3 : [('x4', 1), (None, 0)],
                4 : [('x5', 1), (None, 0)],
            },
        '(x2 = 0) & (x3 = 0)' : {0 : [('x2', 1), (None, 0)],
                                 1 : [('x3', 1), (None, 0)]
            },

        # test multiplication (still linear hypotheses)
        '(2*x2 = 0) & (3.5*x3 - 1)' : {0 : [('x2', 2),(None, 0)],
                                       1 : [('x3', 3.5), (None, -1)] },
        # division
        '(2*x2 = 0) & (x3/3 - 1)' : {0 : [('x2', 2),(None, 0)],
                                       1 : [('x3', 1/3.), (None, -1)] },
        # how to deal with powers? is this still a linear hypothesis?

        # some more complicated ones
        '(2*x1 = 1) & (x2=3) & (x1 + x6 = x2 - 5)' : {
            0 : [('x1', 2), (None, 1)],
            1 : [('x2', 1), (None, 3)],
            2 : [[('x1', 1),('x2', -1),('x6', 1)], (None, -5)],
            },
        '(x1 = x2) & (x4 = 2) & (x5/5 = 1)' : {
            0 : [[('x1', 1), ('x2', -1)], (None, 0)],
            1 : [('x4', 1), (None, 2)],
            2 : [('x5', 1/5.), (None, 1)],
                                        },
        # RHS -> LHS simplification bugs pointed out by Josef
        '(2*x1 = x1)' : [('x1', 1), (None, 0)],
        'x1 + x2 = 2*x2' : [[('x1', 1), ('x2', -1)], (None, 0)],
        'x1 + 2*x2 = x2' : [[('x1', 1), ('x2', 1)], (None, 0)],
        'x1 + 2*x2 = -x2' : [[('x1', 1), ('x2', 3)], (None, 0)],
        # notice the term order of the formula (arbitrary for the result)
        'x1 + 2*x2 + x3 = -x2' : [[('x1', 1), ('x2', 3), ('x3', 1)], (None, 0)],

    }

# this should actually be a parser error
_test_eval_errors = [
        '1 = 1',
        'x1*x2 = 3',
        'x1 ** 2 = 3', #?
        '((x1 = 3) & ((x2 = 3) & (x3 = 3))', # error?
        ' = 1' # empty LHS
        ]

class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self,kw)
        self.__dict__ = self

import numpy.testing as npt
def _do_test_make_matrices(code, result):
    # simulate a model
    model_results = Bunch(model=Bunch(exog_names=result[0]))
    Q, R = make_hypotheses_matrices(model_results, code)
    npt.assert_almost_equal(Q, result[1][0])
    npt.assert_almost_equal(R, result[1][1])

def test_make_matrices():
    for code, result in _test_make_matrices.iteritems():
        _do_test_make_matrices(code, result)


_test_make_matrices = {
        # code : names, result
        '(2*x1 = 1) & (x2=3) & (x1 + x6 = x2 - 5)' :
                (['const', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
                 ([[0, 2, 0, 0, 0, 0, 0],[0,0,1,0,0,0,0],[0,1,-1,0,0,0,1]],
                   [1,3,-5]))
        }

if __name__ == "__main__":
    from parser import parse
    tree = parse('x1+x2=x3')

