"""
Provides a function to open the system browser to either search or go directly
to a function's reference
"""
import webbrowser

from statsmodels.compat.python import urlencode, string_types
from statsmodels import __version__

BASE_URL = 'https://www.statsmodels.org/'


def _generate_url(arg, stable):
    """
    Parse inputs and return a correctly formatted URL or raises ValueError
    if the input is not understandable
    """
    url = BASE_URL
    if stable:
        url += 'stable/'
    else:
        url += 'devel/'

    if arg is None:
        return url
    elif isinstance(arg, string_types):
        url += 'search.html?'
        url += urlencode({'q': arg})
        url += '&check_keywords=yes&area=default'
    else:
        try:
            func = arg
            func_name = func.__name__
            func_module = func.__module__
            if not func_module.startswith('statsmodels.'):
                raise ValueError('Function must be from statsmodels')
            url += 'generated/'
            url += func_module + '.' + func_name + '.html'
        except AttributeError:
            raise ValueError('Input not understood')
    return url


def webdoc(arg=None, stable=None):
    """
    Opens a browser and displays online documentation

    Parameters
    ----------
    arg, optional : string or statsmodels function
        Either a string to search the documentation or a function
    stable, optional : bool
        Flag indicating whether to use the stable documentation (True) or
        the development documentation (False).  If not provided, opens
        the stable documentation if the current version of statsmodels is a
        release

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> sm.webdoc()  # Documention site
    >>> sm.webdoc('glm')  # Search for glm in docs
    >>> sm.webdoc(sm.OLS, stable=False)  # Go to generated help for OLS, devel

    Notes
    -----
    By default, open stable documentation if the current version of statsmodels
    is a release.  Otherwise opens the development documentation.

    Uses the default system browser.
    """
    stable = __version__ if 'dev' not in __version__ else stable
    url_or_error = _generate_url(arg, stable)
    webbrowser.open(url_or_error)
    return None
