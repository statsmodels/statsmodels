#! /usr/bin/env python

"""Mauna Loa Weekly Atmospheric CO2 Data"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = """Mauna Loa Weekly Atmospheric CO2 Data"""
SOURCE      = """
Data obtained from http://cdiac.ornl.gov/trends/co2/sio-keel-flask/sio-keel-flaskmlo_c.html

Obtained on 3/15/2014.

Citation:

Keeling, C.D. and T.P. Whorf. 2004. Atmospheric CO2 concentrations derived from flask air samples at sites in the SIO network. In Trends: A Compendium of Data on Global Change. Carbon Dioxide Information Analysis Center, Oak Ridge National Laboratory, U.S. Department of Energy, Oak Ridge, Tennessee, U.S.A.
"""

DESCRSHORT  = """Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A."""

DESCRLONG   = """
Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.

Period of Record: March 1958 - December 2001

Methods: An Applied Physics Corporation (APC) nondispersive infrared gas analyzer was used to obtain atmospheric CO2 concentrations, based on continuous data (four measurements per hour) from atop intake lines on several towers. Steady data periods of not less than six hours per day are required; if no such six-hour periods are available on any given day, then no data are used that day. Weekly averages were calculated for most weeks throughout the approximately 44 years of record. The continuous data for year 2000 is compared with flask data from the same site in the graphics section."""

#suggested notes
NOTE        = """::

    Number of observations: 2225
    Number of variables: 2
    Variable name definitions:

        date - sample date in YYMMDD format
        co2 - CO2 Concentration ppmv

    The data returned by load_pandas contains the dates as the index.
"""

import numpy as np
from statsmodels.datasets import utils as du
from os.path import dirname, abspath

import pandas as pd

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    names = data.dtype.names
    return du.Dataset(data=data, names=names)


def load_pandas():
    data = load()
    # pandas <= 0.12.0 fails in the to_datetime regex on Python 3
    index = pd.DatetimeIndex(start=data.data['date'][0].decode('utf-8'),
                             periods=len(data.data), format='%Y%m%d',
                             freq='W-SAT')
    dataset = pd.DataFrame(data.data['co2'], index=index, columns=['co2'])
    #NOTE: this is how I got the missing values in co2.csv
    #new_index = pd.DatetimeIndex(start='1958-3-29', end=index[-1],
    #                             freq='W-SAT')
    #data.data = dataset.reindex(new_index)
    data.data = dataset
    return data


def _get_data():
    filepath = dirname(abspath(__file__))
    with open(filepath + '/co2.csv', 'rb') as f:
        data = np.recfromtxt(f, delimiter=",", names=True, dtype=['a8', float])
    return data
