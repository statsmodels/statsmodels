"""
It occured to me that I may have just rewritten some of sympy for this...
"""

from charlton.util import to_unique_tuple
import charlton.builtins
from charlton.eval import EvalEnvironment
from charlton.parse import ParseNode
from parser import parse

def _maybe_simplify_rhs(rhs):
    if not rhs:
        rhs = (None, 0.0)
    else:
        add_scalars = lambda x, y: (None, x[1]+y[1])
        if isinstance(rhs, list):
            rhs = reduce(add_scalars, rhs)
    return rhs

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
        for term in exprs[0]: #TODO:  is the tuple check right
            #if not isinstance(term, tuple) and Evaluator._is_a(float, term):
            if term[0] is None and Evaluator._is_a(float, term[1]):
                rhs += [(term[0], term[1] * -1)]
            else:
                lhs += [term]
    elif exprs[0][0] is None:
        # then we know it's a scalar only on LHS, flip the expression
        # and start over, careful for infinute recursion if two scalars only
        # should be caught earlier though...
        return _maybe_rearrange_terms(exprs[::-1])
    else: # single term on LHS not a scalar
        lhs = exprs[0]

    if len(lhs) == 1: # to be consistent with other single term LHS
        lhs = lhs[0]
    # do RHS
    if isinstance(exprs[1], list):
        for term in exprs[1]:
            if not isinstance(term, tuple) and Evaluator._is_a(float, term):
                rhs += [(term[0], term[1] * -1)]
            else:
                lhs += [(term[0], term[1] * -1)]
    elif exprs[1][0] is not None: # it's a term
        lhs += [(exprs[1][0], exprs[1][1] * -1)]
    else:
        rhs += [exprs[1]]

    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    rhs = _maybe_simplify_rhs(rhs)

    return [lhs, rhs]

def _eval_binary_ampersand(evaluator, tree):
    exprs = [evaluator.eval(arg) for arg in tree.args]
    tests = {}
    for i in range(len(exprs)):
        tests.update({ i : exprs[i]})
    return tests

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
    exprs = [evaluator.eval(arg) for arg in tree.args]
    return IntermediateExpr(False, None, False,
                            exprs[0].terms
                            + exprs[1].terms
                            + _interaction(*exprs).terms)

def _eval_binary_div(evaluator, tree):
    left_expr = evaluator.eval(tree.args[0])
    right_expr = evaluator.eval(tree.args[1])
    terms = list(left_expr.terms)
    _check_interactable(left_expr)
    # Build a single giant combined term for everything on the left:
    left_factors = []
    for term in left_expr.terms:
        left_factors += list(term.factors)
    left_combined_expr = IntermediateExpr(False, None, False,
                                          [Term(left_factors)])
    # Then interact it with everything on the right:
    terms += list(_interaction(left_combined_expr, right_expr).terms)
    return IntermediateExpr(False, None, False, terms)

def _eval_binary_power(evaluator, tree):
    left_expr = evaluator.eval(tree.args[0])
    _check_interactable(left_expr)
    power = -1
    try:
        power = int(tree.args[1])
    except (ValueError, TypeError):
        pass
    if power < 1:
        raise CharltonError("'**' requires a positive integer", tree.args[1])
    all_terms = left_expr.terms
    big_expr = left_expr
    # Small optimization: (a + b)**100 is just the same as (a + b)**2.
    power = min(len(left_expr.terms), power)
    for i in xrange(1, power):
        big_expr = _interaction(left_expr, big_expr)
        all_terms = all_terms + big_expr.terms
    return IntermediateExpr(False, None, False, all_terms)

def _eval_unary_plus(evaluator, tree):
    return evaluator.eval(tree.args[0])

def _eval_unary_minus(evaluator, tree):
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
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
        self.add_op("**", 2, _eval_binary_power)

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

    def eval(self, tree):
        result = None
        if isinstance(tree, str):
            if self._is_a(int, tree) or self._is_a(float, tree):
                result = None, float(tree)
            else:
                # Guess it's a parameter with unitary coefficient
                result = tree, 1 # don't convert to string? use origin anywhere?
        elif isinstance(tree, tuple): # got a param, coeff pair
            result = tree
        else:
            assert isinstance(tree, ParseNode)
            key = (tree.op.token, len(tree.args))
            if key not in self._evaluators:
                raise CharltonError("I don't know how to evaluate "
                                    "this '%s' operator" % (tree.op.token,),
                                    tree.op)
            result = self._evaluators[key](self, tree)
        return result

def make_hypotheses_matrices(model_results, code, depth=0):
    tree = parse(code)
    #we shouldn't need any of the builtins for testing
    #eval_env.add_outer_namespace(charlton.builtins.builtins)
    #do we even need an eval environment?
    test_desc = Evaluator({}).eval(tree)
    lhs = test_desc[0]
    rhs = test_desc[1]
    exog_names = model_results.model

    #desc = evaluate_tree(tree, eval_env)
    R,Q = make_hypotheses(desc)
    return R,Q

def _do_eval_test(test_cases):
    for code, expected in test_cases.iteritems():
        tree = parse(code)
        actual = Evaluator({}).eval(tree)
        length = len(actual)
        if length > 1:
            for i in range(length):
                assert actual[i] == expected[i]
        else:
            assert actual == expected

def test_eval():
    _do_eval_test(_test_eval)

_test_eval = {
        # some basic tests

        # results are parameter, it's coefficient plus the RHS term
        'x2 = 0' : [('x2', 1), (None, 0)],
        '0 = x2' : [('x2', 1), (None, 0)],
        'x1 + x2 = 0' : [[('x1', 1), ('x2', 1)], (None, 0)],
        '0 = x1 + x2' : [[('x1', 1), ('x2', 1)], (None, 0)],
        'x1 - x2 = 0' : [[('x1', 1), ('x2', -1)], (None, 0)],
        '-x2 = 1' : [('x2', -1), (None, 1)],
        'x1 + x2 = x3' : [[('x1', 1), ('x2', 1), ('x3', -1)], (None, 0)],
        # next takes advantage of sympy
        '-(x1 + x2) = 1' : [[('x1', -1), ('x2', -1)], (None, 1)],
        'x1 + x2 = x3 + x4' : [[('x1', 1), ('x2', 1),
                                ('x3', -1), ('x4', -1)], (None, 0)],
        'x1 + 5 = 3' : [('x1', 1), (None, -2)],
        'x1 + 3 - 2' : [('x1', 1), (None, -1)],
        # multiple hypotheses
        '(x2 = 0) & (x3 = 0)' : {0 : [('x2', 1), (None, 0)],
                                 1 : [('x3', 1), (None, 0)]
            },

        # test multiplication (still linear hypotheses)

    }

# this should actually be a parser error
_test_eval_errors = [
        '1 = 1',
        ]

if __name__ == "__main__":
    from parser import parse
    tree = parse('x1+x2=x3')

