"""For top level testing of the statsmodels package"""

from numpy.testing import NumpyTest

def additional_tests():
# note from copied tester.py
# does this guarantee that the package is the one in dev trunk and not
# scikits.foo installed somewhere else?
    import scikits.statsmodels
    np = NumpyTest(scikits.statsmodels)
    return np._test_suite_from_all_tests(np.package, level = 10, verbosity = 1)

