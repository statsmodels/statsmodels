"""
Provides a function to open the system browser to either search or go directly
to a function's reference
"""
import webbrowser
import urllib

BASE_URL = 'http://statsmodels.sourceforge.net/stable/'


def doc(arg=None):
    """
    Parameters
    ----------
    arg, optional : string or statsmodels function
        Either a string to search the documentation or a function

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> sm.doc()  # Documention site
    >>> sm.doc('glm')  # Search for glm in docs
    >>> sm.doc(sm.OLS)  # Go to generated help for OLS

    Notes
    -----
    Opens in the system default browser
    """
    if arg is None:
        url = BASE_URL
    elif type(arg) is str:
        url = BASE_URL + 'search.html?'
        url += urllib.urlencode({'q': arg})
        url += '&check_keywords=yes&area=default'
    else:
        try:
            func = arg
            func_name = func.__name__
            func_module = func.__module__
            url = BASE_URL + 'generated/'
            url += func_module + '.' + func_name + '.html'
        except:
            raise ValueError('Input not understood')
    webbrowser.open(url)
    return None



