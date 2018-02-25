__all__ = [
    'pvalue_to_stars'
]


def pvalue_to_stars(p):
    """
    Represent a p-value's significance with stars.

    Parameters
    ----------
    p : float

    Returns
    -------
    str
        Representation of the p-value's significance.

    """
    if not 0 <= p <= 1:
        raise ValueError('p must be in range [0, 1]')

    if p <= 0.0001:
        return '****'
    elif p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return ''
