# -*- coding: utf-8 -*-
# Last Change: Wed Jun 24 06:00 PM 2009

# The code and descriptive text is copyrighted and offered under the terms of
# the BSD License from the authors; see below. However, the actual dataset may
# have a different origin and intellectual property status. See the SOURCE and
# COPYRIGHT variables for this information.

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     * Neither the author nor the names of any contributors may be used
#       to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

"""US Capital Punishment dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with expressed permission from the original author,
who retains all rights."""
TITLE       = "US Capital Punishment Dataset"
SOURCE      = """
Jeff Gill's `Generalized Linear Models: A Unifited Approach

http://jgill.wustl.edu/research/books.html
"""

DESCRSHORT  = """Number of state executions in 1997"""

DESCRLONG   = """
This data describes the number of times capital punishment is implemented
at the state level for the year 1997.  The outcome variable is the number of
executions.  There were executions in 17 states.
Included in the data are explanatory variables for median per capita income
in dollars, the percent of the population classified as living in poverty,
the percent of Black citizens in the population, the rate of violent
crimes per 100,000 residents for 1996, a dummy variable indicating
whether the state is in the South, and (an estimate of) the proportion
of the population with a college degree of some kind.

The original source files are included in /cpunish/src/

/cpunish/cpunish.csv contains the cleaned data for the example
in a comma-delimited file
"""

NOTE        = """
Number of Observations: 17.
Number of Variables: 7
Variable name definitions:
    EXECUTIONS: Executions in 1996
    INCOME: Median per capita income in 1996 dollars
    PERPOVERTY: Percent of the population classified as living in poverty
    PERBLACK: Percent of black citizens in the population
    VC100k96: Rate of violent crimes per 100,00 residents for 1996
    SOUTH: SOUTH == 1 indicates a state in the South
    DEGREE: An esimate of the proportion of the state population with a
        college degree of some kind

State names are included in the data file, though not returned by load.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the cpunish data and return a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(filepath + '/cpunish.csv', delimiter=",",
            names=True, dtype=float, usecols=(1,2,3,4,5,6,7))
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name=endog_name, exog_name=exog_name)
    return dataset
