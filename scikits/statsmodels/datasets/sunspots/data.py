#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Last Change: Tue Jul 17 05:00 PM 2007 J

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

"""Yearly sunspots data 1700-2008"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This data is public domain."""
TITLE       = ""
SOURCE      = """
http://www.ngdc.noaa.gov/stp/SOLAR/ftpsunspotnumber.html

The original dataset contains monthly data on sunspot activity in the file
./src/sunspots_yearly.dat.  There is also sunspots_monthly.dat.
"""

DESCRSHORT  = """Yearly (1700-2008) data on sunspots from the National
Geophysical Data Center."""

DESCRLONG   = DESCRSHORT

NOTE        = """
Number of Observations: 309 (Annual 1700 - 2008)
Number of Variables: 1
Variable name definitions:
    SUNACTIVITY : Number of sunspots for each year

The data file contains a 'YEAR' variable that is not returned by load.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the yearly sunspot data and returns a data class.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    This dataset only contains data for one variable, so the attributes
    data, raw_data, and endog are all the same variable.  There is no exog
    attribute defined.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/sunspots.csv', 'rb'), delimiter=",",
            names=True, dtype=float, usecols=(1))
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names
    dataset = Dataset(data=data, names=names, endog=endog,
            endog_name=endog_name)
    return dataset

