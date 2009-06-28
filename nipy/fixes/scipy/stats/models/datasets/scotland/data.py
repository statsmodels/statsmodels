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

"""Taxation Powers Vote for the Scottish Parliament 1997 dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Permission granted by the author."""
TITLE       = "Taxation Powers Vote for the Scottish Parliamant 1997"
SOURCE      = """
http://jgill.wustl.edu/research/books.html

Dr. Jeff Gill
Department of Political Science
One Brookings Drive, Seigle L079
Washington University
St. Louis, MO 63130-4899
"""

DESCRSHORT  = """Taxation Powers' Yes Vote for Scottish Parliamanet-1997"""

DESCRLONG   = """
This data describes the proportion of voters who voted Yes to grant the
Scottish Parliament taxation powers.  The data are divided into 32
council districts.  This model's explanatory variables include the amount
of council tax collected in pounds sterling as of April 1997 per two
adults before adjustments, the female percentage of total claims for
unemployment benefits as of January, 1998, the standardized mortality rate
(UK is 100), the percentage of labor force participation, regional GDP, the percentage
of children aged 5 to 15, and an interaction term between female unemployment and
the council tax.

The original source files and variable information are included in
/scotland/src/

/scotland/scotland.csv contains the cleaned subset of the original data
for the example in a comma-delimited file
"""

NOTE        = """
Number of Instances: 32

Number of Attributes: 5

Missing Attribute Values: None

Council district names are included in the data file, though not returned.
"""

import numpy as np

class load():
    """load the scotland data and returns a data class.

    :returns:
        data instance:
            a class of the data with array attrbutes 'endog' and 'exog'
    """
    def __init__(self):
        from scotvote import __dict__, names
        self._names = names
        self._d = __dict__
        self.endog = np.array(self._d[self._names[1]], dtype=np.float)
        self.exog = np.column_stack(self._d[i] \
                    for i in self._names[2:]).astype(np.float)
