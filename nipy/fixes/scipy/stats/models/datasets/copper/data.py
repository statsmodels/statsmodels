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

"""World Copper Prices 1951-1975 dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Permission granted by the author."""
TITLE       = "World Copper Market 1951-1975 Dataset"
SOURCE      = """
http://jgill.wustl.edu/research/books.html

Dr. Jeff Gill
Department of Political Science
One Brookings Drive, Seigle L079
Washington University
St. Louis, MO 63130-4899
"""

DESCRSHORT  = """World Copper Market 1951-1975"""

DESCRLONG   = """
This data describes the world copper market from 1951 to 1975.  The outcome
variable (of a 2 stage estimation) is the world consumption of copper for
the 25 years.  The variables included in the dataset are the world consumption
of copper in 1000 metric tons, the constant dollar adjusted price of copper,
the price of a substitute, aluminum, an index of real per capita income
base 1970, an annual measure of manufacturer inventory change, and a time
trend.

The original source files are included in /copper/src/

/copper/copper.csv contains the cleaned data for the example
in a comma-delimited file
"""

NOTE        = """
Number of Instances: 17.

Number of Attributes: 6

Missing Attribute Values: None

Years are included in the data file, though not returned.
"""

import numpy as np

class load():
    """load the copper data and returns a data class.

    :returns:
        data instance:
            a class of the data with array attrbutes 'endog' and 'exog'
    """
    _endog = None
    _exog = None
    def __init__(self):
        from copper import __dict__, names
        self._names = names
        self._d = __dict__

    @property
    def endog(self):
        if self._endog is None:
            self.endog = np.array(self._d[self._names[1]], dtype=np.float)
        return self.endog

    @property
    def exog(self):
        if self._exog is None:
            self.exog = np.column_stack(self._d[i] \
                    for i in self._names[2:]).astype(np.float)
        return self.exog

