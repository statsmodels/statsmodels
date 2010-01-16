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

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'Load']

"""Grunfeld Investment Data"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = "Grunfeld Investment Data"
SOURCE      = """
This is the well-known Grunfeld (1950) Investment Data.

The source for the data was the original 11-firm data set from Grunfeld's Ph.D.
thesis recreated by Kleiber and Zeileis (2008) "The Grunfeld Data at 50".
The data can be found here.
http://statmath.wu-wien.ac.at/~zeileis/grunfeld/

For a note on the many version of the Grunfeld data circulating see:
http://www.stanford.edu/~clint/bench/grunfeld.htm
"""

DESCRSHORT  = """Grunfeld (1950) Investment Data for 11 U.S. Firms."""

DESCRLONG   = DESCRSHORT

NOTE        = """
Number of observations: 220 (20 years for 11 firms)

Variables
----------
invest - Gross investment in 1947 dollars
value - Market value as of Dec. 31 in 1947 dollars
capital - Stock of plant and equipment in 1947 dollars
firm - General Motors, US Steel, General Electric, Chrysler, Atlantic Refining,
        IBM, Union Oil, Westinghouse, Goodyear, Diamond Match, American Steel
year - 1935 - 1954
"""

import numpy as np

class Load():
    """Loads the Grunfeld data and returns a data class.

    Returns
    Load instance:
        a class of the data with array attrbutes 'endog' and 'exog'

    Notes
    -----
    `endog` is a structured array for the Grunfled data since it contains
    a float variable.
    """
    def __init__(self):
        from grunfeld import __dict__, names
        self._names = [_.lower() for _ in names]
        self._d = __dict__
        self.endog = np.array(self._d[self._names[0].upper()], dtype=np.float)
        dt = np.dtype([('value',float),('capital',float),('firm', 'a17'),
            ('year',float)])
        self.exog = np.zeros(220, dtype=dt)
        for i in dt.names:
            self.exog[i] = np.array(self._d[i.upper()]).astype(dt[i])
