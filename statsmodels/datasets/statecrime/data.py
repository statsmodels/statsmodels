#! /usr/bin/env python

"""Statewide Crime Data"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Public domain."""
TITLE       = """Statewide Crime Data 2009"""
SOURCE      = """
All data is for 2009 and was obtained from the American Statistical Abstracts except as indicated below.
"""

DESCRSHORT  = """State crime data 2009"""

DESCRLONG   = DESCRSHORT

#suggested notes
NOTE        = """::

    Number of observations: 51
    Number of variables: 8
    Variable name definitions:

    state
        All 50 states plus DC.
    violent
        Rate of violent crimes / 100,000 population. Includes murder, forcible
        rape, robbery, and aggravated assault. Numbers for Illinois and
        Minnesota do not include forcible rapes. Footnote included with the
        American Statistical Abstract table reads:
        "The data collection methodology for the offense of forcible
        rape used by the Illinois and the Minnesota state Uniform Crime
        Reporting (UCR) Programs (with the exception of Rockford, Illinois,
        and Minneapolis and St. Paul, Minnesota) does not comply with
        national UCR guidelines. Consequently, their state figures for
        forcible rape and violent crime (of which forcible rape is a part)
        are not published in this table."
    murder
        Rate of murders / 100,000 population.
    hs_grad
        Precent of population having graduated from high school or higher.
    poverty
        % of individuals below the poverty line
    white
        Percent of population that is one race - white only. From 2009 American
        Community Survey
    single
        Calculated from 2009 1-year American Community Survey obtained obtained
        from Census. Variable is Male householder, no wife present, family
        household combined with Female household, no husband prsent, family
        household, divided by the total number of Family households.
    urban
        % of population in Urbanized Areas as of 2010 Census. Urbanized
        Areas are area of 50,000 or more people."""

import numpy as np
from statsmodels.datasets import utils as du
from os.path import dirname, abspath

def load():
    """
    Load the statecrime data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    ##### SET THE INDICES #####
    #NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray(data, endog_idx=2, exog_idx=[7, 4, 3, 5],
                               dtype=float)

def load_pandas():
    data = _get_data()
    ##### SET THE INDICES #####
    #NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray_pandas(data, endog_idx=2, exog_idx=[7,4,3,5],
                                      dtype=float, index_idx=0)

def _get_data():
    filepath = dirname(abspath(__file__))
    ##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    with open(filepath + '/statecrime.csv', 'rb') as f:
        try:
            data = np.recfromtxt(f, delimiter=",", names=True,
                                 dtype=None, encoding='utf-8')
        except TypeError:
            data = np.recfromtxt(f, delimiter=",", names=True, dtype=None)
    return data
