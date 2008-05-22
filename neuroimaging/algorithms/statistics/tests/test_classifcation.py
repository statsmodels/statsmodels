from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.algorithms.statistics.classification')


if __name__ == '__main__':
    NumpyTest.run()
