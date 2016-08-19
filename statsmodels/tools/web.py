"""
Provides a function to open the system browser to either search or go directly
to a function's reference
"""
import webbrowser

from statsmodels.compat.python import urlencode
from statsmodels.version import release

BASE_URL = 'http://www.statsmodels.org/'


def _generate_url(arg, stable):
    """
    Parse inputs and return a correctly formatted URL or an error if the input
    is not understandable
    """
    url = BASE_URL
    if stable:
        url += 'stable/'
    else:
        url += 'devel/'

    if arg is None:
        return url
    elif type(arg) is str:
        url += 'search.html?'
        url += urlencode({'q': arg})
        url += '&check_keywords=yes&area=default'
    else:
        try:
            func = arg
            func_name = func.__name__
            func_module = func.__module__
            if not func_module.startswith('statsmodels.'):
                return ValueError('Function must be from statsmodels')
            url += 'generated/'
            url += func_module + '.' + func_name + '.html'
        except:
            return ValueError('Input not understood')
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
    stable = release if stable is None else stable
    url_or_error = _generate_url(arg, stable)
    if isinstance(url_or_error, ValueError):
        raise url_or_error
    webbrowser.open(url_or_error)
    return None
