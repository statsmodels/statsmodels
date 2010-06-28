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

"""Taxation Powers Vote for the Scottish Parliament 1997 dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with expressed permission from the original author,
who retains all rights."""
TITLE       = "Taxation Powers Vote for the Scottish Parliamant 1997"
SOURCE      = """
Jeff Gill's `Generalized Linear Models: A Unifited Approach

http://jgill.wustl.edu/research/books.html
"""
DESCRSHORT  = """Taxation Powers' Yes Vote for Scottish Parliamanet-1997"""

DESCRLONG   = """
This data is based on the example in Gill and describes the proportion of
voters who voted Yes to grant the Scottish Parliament taxation powers.
The data are divided into 32 council districts.  This example's explanatory
variables include the amount of council tax collected in pounds sterling as
of April 1997 per two adults before adjustments, the female percentage of
total claims for unemployment benefits as of January, 1998, the standardized
mortality rate (UK is 100), the percentage of labor force participation,
regional GDP, the percentage of children aged 5 to 15, and an interaction term
between female unemployment and the council tax.

The original source files and variable information are included in
/scotland/src/
"""

NOTE        = """
Number of Observations: 32 (1 for each Scottish district)
Number of Variables: 8
Variable name definitions:
    YES : Proportion voting yes to granting taxation powers to the Scottish
        pariliament.
    COUTAX : Amount of council tax collected in pounds steling as of April '97
    UNEMPF : Female percentage of total unemployment benefits claims as of
        January 1998
    MOR : The standardized mortality rate (UK is 100)
    ACT : Labor force participation (Short for active)
    GDP : GDP per county
    AGE : Percentage of children aged 5 to 15 in the county
    COUTAX_FEMALEUNEMP : Interaction between COUTAX and UNEMPF

Council district names are included in the data file, though are not returned
by load.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the Scotvote data and returns a Dataset instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(filepath + '/scotvote.csv', delimiter=",",
            names=True, dtype=float, usecols=(1,2,3,4,5,6,7,8))
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset

