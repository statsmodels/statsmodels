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

"""Longley dataset"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = "Longley dataset"
SOURCE      = """
The classic 1967 Longley Data

http://www.itl.nist.gov/div898/strd/lls/data/Longley.shtml

Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the Electronic
    Comptuer from the Point of View of the User."  Journal of the American
    Statistical Association.  62.319, 819-41.
"""

DESCRSHORT  = """"""

DESCRLONG   = """The Longley dataset contains various US macroeconomic
variables that are known to be highly collinear.  It has been used to appraise
the accuracy of least squares routines."""

NOTE        = """
Number of Observations: 16
Number of Variables: 6
Variable name definitions: TOTEMP : Total Employment
                           GNPDEFL : GNP deflator
                           GNP : GNP
                           UNEMP : Number of unemployed
                           ARMED : Size of armed forces
                           POP : Population
                           YEAR : Year (1947 - 1962)
"""

from numpy import recfromtxt, array, column_stack
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the Longley data and return a Dataset class.

    Returns
    -------
    Dataset instance
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(filepath+'/longley.csv', delimiter=",", names=True,
            dtype=float, usecols=(1,2,3,4,5,6,7))
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name=endog_name, exog_name = exog_name)
    return dataset
