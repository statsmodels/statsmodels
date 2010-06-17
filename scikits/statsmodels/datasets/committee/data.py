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

"""First 100 days of the US House 1995 dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with expressed permission from the original author,
who retains all rights."""
TITLE       = "First 100 days of 1995 US House Dataset"
SOURCE      = """
Jeff Gill's `Generalized Linear Models: A Unifited Approach`

http://jgill.wustl.edu/research/books.html
"""

DESCRSHORT  = """Number of bill assignments in the 104th House in 1995"""

DESCRLONG   = """
The example in Gill, seeks to explain the number of bill assignments in the
first 100 days of the US' 104th House of Representatives.  The response
variable is the number of bill assignments in the first 100 days over 20
Committees.  The explanatory variables in the example are the number of
assignments in the first 100 days of the 103rd House, the number of members on
the committee, the number of subcommittees, the log of the number of staff
assigned to the committee, a dummy variable indicating whether
the committee is a high prestige committee, and an interaction term between
the number of subcommittees and the log of the staff size.

The data returned by load are not yet cleaned to represent the above example.

The original source files are included in /committee/src/
"""

NOTE = """
Number of Observations: 20
Number of Variables: 6
Variable name definitions:
    BILLS104 : Number of bill assignments in the first 100 days of the 104th
        House of Representatives.
    SIZE : Number of members on the committee.
    SUBS : Number of subcommittees.
    STAFF : Number of staff members assigned to the committee.
    PRESTIGE : PRESTIGE == 1 is a high prestige committee.
    BILLS103 : Number of bill assignments in the first 100 days of the 103rd
        House of Representatives.

Committee names are included as a variable in the data file though not
returned by load.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """Load the committee data and returns a data class.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(filepath + '/committee.csv', delimiter=",",
            names=True, dtype=float, usecols=(1,2,3,4,5,6))

    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset


