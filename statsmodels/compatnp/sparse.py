from scipy.sparse import coo_matrix, issparse, bmat

#imported from scipy-dev
def block_diag(mats, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix from provided matrices.

    Parameters
    ----------
    A, B, ... : sequence of matrices
        Input matrices.
    format : str, optional
        The sparse format of the result (e.g. "csr").  If not given, the matrix
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output matrix.  If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    res : sparse matrix

    See Also
    --------
    bmat, diags

    Examples
    --------
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5], [6]])
    >>> C = coo_matrix([[7]])
    >>> block_diag((A, B, C)).todense()
    matrix([[1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 6, 0],
            [0, 0, 0, 7]])

    """
    nmat = len(mats)
    rows = []
    for ia, a in enumerate(mats):
        row = [None]*nmat
        if issparse(a):
            row[ia] = a
        else:
            row[ia] = coo_matrix(a)
        rows.append(row)
    return bmat(rows, format=format, dtype=dtype)

