import pytest


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true",
                     help="skip slow tests")
    parser.addoption("--only-slow", action="store_true",
                     help="run only slow tests")
    parser.addoption("--skip-examples", action="store_true",
                     help="skip tests of examples")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and item.config.getoption("--skip-slow"):
        pytest.skip("skipping due to --skip-slow")

    if 'slow' not in item.keywords and item.config.getoption("--only-slow"):
        pytest.skip("skipping due to --only-slow")

    if 'example' in item.keywords and item.config.getoption("--skip-examples"):
        pytest.skip("skipping due to --skip-examples")


def pytest_configure(config):
    try:
        import matplotlib
        matplotlib.use('agg')
    except ImportError:
        pass


@pytest.fixture()
def close_figures():
    """
    Fixture that closes all figures after a test function has completed

    Returns
    -------
    closer : callable
        Function that will close all figures when called.

    Notes
    -----
    Used by passing as an argument to the function that produces a plot,
    for example

    def test_some_plot(close_figures):
        <test code>

    If a function creates many figures, then these can be destroyed within a
    test function by calling close_figures to ensure that the number of
    figures does not become too large.

    def test_many_plots(close_figures):
        for i in range(100000):
            plt.plot(x,y)
            close_figures()
    """
    try:
        import matplotlib.pyplot

        def close():
            matplotlib.pyplot.close('all')

    except ImportError:
        def close():
            pass

    yield close
    close()
