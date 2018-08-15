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


@pytest.fixture()
def close_figures():
    """
    Fixture that closes all figures after a test function has completed

    Notes
    -----
    Used by passing as an argument to the function that produces a plot,
    for example

    def test_some_plot(close_figures):
        <test code>
    """
    yield None
    try:
        from matplotlib.pyplot import close
        close('all')
    except ImportError:
        pass
