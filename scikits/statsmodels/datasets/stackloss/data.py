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

"""Stack loss data."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain. """
TITLE       = "Brownlee's classic stack loss plant data"
SOURCE      = """
Brownlee, K. A. (1965), "Statistical Theory and Methodology in
Science and Engineering", 2nd edition, New York:Wiley.
"""

DESCRSHORT  = """Stack loss plant data of Brownlee (1965)"""

DESCRLONG   = """The stack loss plant data of Brownlee (1965) contains
21 days of measurements from a plant's oxidation of ammonia to nitric acid.
The nitric oxide pollutants are captured in an absorption tower."""

NOTE        = """
Number of Observations: 21
Number of Variables: 4
Variable name definitions:
    STACKLOSS : 10 times the percentage of ammonia going into the plant that
        escapes from the absoroption column
    AIRFLOW : Rate of operation of the plant
    WATERTEMP : Cooling water temperature in the absorption tower
    ACIDCONC : Acid concentration of circulating acid minus 50 times 10.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the stack loss data and returns a Dataset class instance.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/stackloss.csv',"rb"), delimiter=",",
            names=True, dtype=float)
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
