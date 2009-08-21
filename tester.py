"""For top level testing of the statsmodels package"""

from numpy.testing import Tester

def additional_tests():
# note from copied tester.py
# does this guarantee that the package is the one in dev trunk and not
# scikits.foo installed somewhere else?
    import scikits.statsmodels
    test = Tester().test
    return test
