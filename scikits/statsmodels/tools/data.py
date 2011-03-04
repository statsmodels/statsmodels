"""
Compatibility tools for various data structure inputs
"""

import numpy as np

def have_pandas():
    try:
        import pandas
        return True
    except ImportError:
        return False
    except Exception:
        return False

def is_data_frame(obj):
    if not have_pandas():
        return False

    import pandas as pn

    return isinstance(obj, pn.DataFrame)

def is_structured_ndarray(obj):
    return isinstance(obj, np.ndarray) and obj.dtype.names is not None

def interpret_data(data, colnames=None, rownames=None):
    """
    Convert passed data structure to form required by estimation classes

    Parameters
    ----------
    data : ndarray-like
    colnames : sequence or None
        May be part of data structure
    rownames : sequence or None

    Returns
    -------
    (values, colnames, rownames) : (homogeneous ndarray, list)
    """
    if isinstance(data, np.ndarray):
        if is_structured_ndarray(data):
            if colnames is None:
                colnames = data.dtype.names
            values = struct_to_ndarray(data)
        else:
            values = data

        if colnames is None:
            colnames = ['Y_%d' % i for i in range(values.shape[1])]
    elif is_data_frame(data):
        # XXX: hack
        data = data.dropIncompleteRows()
        values = data.values
        colnames = data.columns
        rownames = data.index
    else: # pragma: no cover
        raise Exception('cannot handle other input types at the moment')

    if not isinstance(colnames, list):
        colnames = list(colnames)

    # sanity check
    if len(colnames) != values.shape[1]:
        raise ValueError('length of colnames does not match number '
                         'of columns in data')

    if rownames is not None and len(rownames) != len(values):
        raise ValueError('length of rownames does not match number '
                         'of rows in data')

    return values, colnames, rownames

def struct_to_ndarray(arr):
    return arr.view((float, len(arr.dtype.names)))
