#! /usr/bin/env python

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

"""Name of dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """E.g., This is public domain."""
TITLE       = """Title of the dataset"""
SOURCE      = """
This section should provide a link to the original dataset if possible and
attribution and correspondance information for the dataset's original author
if so desired.
"""

DESCRSHORT  = """A short description."""

DESCRLONG   = """A longer description of the dataset."""

#suggested notes
NOTE        = """
Number of observations:
Number of variables:
Variable name definitions:

Any other useful information that does not fit into the above categories.
"""

import numpy as np
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    data = np.recfromtxt(open(filepath + '/DatasetName.csv', 'rb'), delimiter=",",
            names=True, dtype=float)
    names = list(data.dtype.names)
##### SET THE INDEX #####
    endog = np.array(data[names[0]], dtype=float)
    endog_name = names[0]
##### SET THE INDEX #####
    exog = np.column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
