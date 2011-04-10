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

"""US Macroeconomic data."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = ""
SOURCE      = """
Compiled by Skipper Seabold. All data are from the Federal Reserve Bank of St.
Louis [1] except the unemployment rate which was taken from the National
Bureau of Labor Statistics [2].

[1] Data Source: FRED, Federal Reserve Economic Data, Federal Reserve Bank of
    St. Louis; http://research.stlouisfed.org/fred2/; accessed December 15,
    2009.

[2] Data Source: Bureau of Labor Statistics, U.S. Department of Labor;
    http://www.bls.gov/data/; accessed December 15, 2009.
"""

DESCRSHORT  = """US Macroeconomic Data for 1959Q1 - 2009Q3"""

DESCRLONG   = DESCRSHORT

NOTE        = """
Number of Observations: 203
Number of Variables: 14
Variable name definitions:
    year      - 1959q1 - 2009q3
    quarter   - 1-4
    realgdp   - Real gross domestic product (Bil. of chained 2005 US$,
                seasonally adjusted annual rate)
    realcons  - Real personal consumption expenditures (Bil. of chained 2005
                US$,
                seasonally adjusted annual rate)
    realinv   - Real gross private domestic investment (Bil. of chained 2005
                US$, seasonally adjusted annual rate)
    realgovt  - Real federal consumption expenditures & gross investment
                (Bil. of chained 2005 US$, seasonally adjusted annual rate)
    realdpi   - Real gross private domestic investment (Bil. of chained 2005
                US$, seasonally adjusted annual rate)
    cpi       - End of the quarter consumer price index for all urban
                consumers: all items (1982-84 = 100, seasonally adjusted).
    m1        - End of the quarter M1 nominal money stock (Seasonally adjusted)
    tbilrate  - Quarterly monthly average of the monthly 3-month treasury bill:
                secondary market rate
    unemp     - Seasonally adjusted unemployment rate (%)
    pop       - End of the quarter total population: all ages incl. armed
                forces over seas
    infl      - Inflation rate (ln(cpi_{t}/cpi_{t-1}) * 400)
    realint   - Real interest rate (tbilrate - infl)
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the US macro data and return a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The macrodata Dataset instance does not contain endog and exog attributes.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/macrodata.csv', 'rb'), delimiter=",", names=True,
            dtype=float)
    names = data.dtype.names
    dataset = Dataset(data=data, names=names)
    return dataset
