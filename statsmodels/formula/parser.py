"""
A parser intended to be used for testing linear and non-linear hypotheses.
This is based heavily on the Charlton parser for "formulas." The big difference
is that it allows for & and = to have lower precedence rather than ~.
"""
import re
from tokenize import ENDMARKER

from charlton.parse import (Operator, _ParseContext, TokenSource, ParseNode,
                            _read_noun_context, _read_op_context, _run_op)
from charlton.tokens import TokenSource

import sympy

# match and by itself
match_and = re.compile('(?<!\w)and(?!\w)')

def _clean_formula(code):
    """
    This uses sympy to simplify some algebraic expressions.
    """
    hypotheses = code.split('&')
    for i in range(len(hypotheses)):
        expr = hypotheses[i].split('=')
        assert len(expr) <= 2
        for j in range(len(expr)):
            try:
                expr[j] = str(sympy.sympify(expr[j]))
            except:
                pass
        hypotheses[i] = ' = '.join(expr)
    return '&'.join(hypotheses)

#NOTE: the only difference between this and charlton.parse.parse is that
# our default operators are different and we return None for an empty code
# parse.parse could be made to be extensible?
def parse(code, extra_operators=[]):
    code = code.replace("\n", " ").strip()
    code = code.replace("==", "=")
    # replace and by &
    code = re.sub(match_and, '&', code)

    if not code:
        return

    code = _clean_formula(code)
    token_source = TokenSource(code)

    for op in extra_operators:
        if op.precedence < 0:
            raise ValueError("All operators must have precedence >= 0")

    all_op_list = _stat_operators + extra_operators
    unary_ops = {}
    binary_ops = {}
    for op in all_op_list:
        if op.arity == 1:
            unary_ops[op.token] = op
        elif op.arity == 2:
            binary_ops[op.token] = op
        else:
            raise ValueError("Operators must be unary or binary")

    c = _ParseContext(unary_ops, binary_ops)
    want_noun = True
    while True:
        if want_noun:
            want_noun = _read_noun_context(token_source, c)
        else:
            if token_source.peek()[0] == ENDMARKER:
                break
            want_noun = _read_op_context(token_source, c)

    while c.op_stack:
        if c.op_stack[-1].token == "(":
            raise CharltonError("Unmatched '('", c.op_stack[-1])
        _run_op(c)

    assert len(c.noun_stack) == 1
    tree = c.noun_stack.pop()
    if not isinstance(tree, ParseNode) or tree.op.token not in ["=", "&"]:
    # & has a lower precedence than = because it can be multiple tests
        tree = ParseNode(unary_ops["="], [tree], tree.origin)
    return tree


_stat_operators = [
    Operator("=", 2, -100),
    Operator("=", 1, -100),

    Operator("&", 2, -200),
    Operator("+", 2, 100),
    Operator("-", 2, 100),
    Operator("*", 2, 200),
    Operator(":", 2, 300),
    Operator("**", 2, 500),

# Do we need any unary operators?
    Operator("+", 1, 100),
    Operator("-", 1, 100),
]

###### TESTS #########

from charlton.parse import _compare_trees

def _do_parse_test(test_cases, extra_operators):
    for code, expected in test_cases.iteritems():
        actual = parse(code, extra_operators=extra_operators)
        print repr(code), repr(expected)
        print actual
        _compare_trees(actual, expected)

def test_parse():
    _do_parse_test(_parser_tests, [])

_parser_tests = {
    "x1 = 1" : ["=","x1","1"],
    "x1 == 1" : ["=","x1","1"],
    "x1" : ["=","x1"],
    "x1 = x2" : ["=", "x1", "x2"],
    "x1 + x2 = x3" : ["=", ["+", "x1", "x2"], "x3"],
    "2*x1 + 2*x2" : ["=", ["+", ["*", "2", "x1"], ["*", "2","x2"]]],
    "2*(x1 + x2)" : ["=", ["+", ["*", "2", "x1"], ["*", "2","x2"]]],
    #before sympy cleaning
    #"2*(x2 + x3) + 4*x4 = 0" : ["=", ["+", ["*", "2", ["+", "x2", "x3"]],
    #                                            ["*", "4","x4"]], "0"],
    #new result
    #code simplifies to 2*x2 + 2*x3 + 4*x4 = 0
    "2*(x2 + x3) + 4*x4 = 0" : ["=", ["+", ["+", ["*", "2", "x2"],
                                                 ["*", "2", "x3"]],
                                                 ["*", "4", "x4"]] , "0"],
        # Multiple Tests
    "(x2 = 0) & (x3 = 0)" : ["&", ["=", "x2", "0"], ["=", "x3", "0"]],
    "(x2 = 0) and (x3 = 0)" : ["&", ["=", "x2", "0"], ["=", "x3", "0"]],
    "(x2 = 0) and (mranderson = 0)" : ["&", ["=", "x2", "0"],
                                       ["=", "mranderson", "0"]],
    "(x2 = 0) & (x3 = 0) & (x4 = 0)" : ["&", ["&", ["=", "x2", "0"],
                                                   ["=", "x3", "0"]],
                                              ["=", "x4", "0"]],
    # Tests in a Multiple Equation Environment
    "x2[2] = 0" : ["=", "x2[2]", "0"], # x2 in 2nd equation = 0

    # Tests that sympy is simplifying things
    "-(3 - (x3 - 5)) = 2" : ["=", ["-", "x3", "8"], "2"]
}
