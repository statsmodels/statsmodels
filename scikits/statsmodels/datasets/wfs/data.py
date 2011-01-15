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

"""World Fertility Survey: Fiji"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Available for use in academic research.  See SOURCE."""
TITLE       = """Title of the dataset"""
SOURCE      = """
The source data was obtained from Germán Rodríguez's web site at Princeton
http://data.princeton.edu/wws509/datasets/#ceb, with the following refernce.

Little, R. J. A. (1978). Generalized Linear Models for Cross-Classified Data
    from the WFS. World Fertility Survey Technical Bulletins, Number 5.

It originally comes from the World Fertility Survey for Fiji
http://opr.princeton.edu/archive/wfs/fj.aspx.

The terms of use for the original dataset are:

Data may be used for academic research, provided that credit is given in any
publication resulting from the research to the agency that conducted the
survey and that two copies of any publication are sent to:

 	Mr. Naibuku Navunisaravi
    Government Statistician
    Bureau of Statistics
    Government Buildings
    P.O. Box 2221
    Suva
    Fiji
"""

DESCRSHORT  = """Fiji Fertility Survey"""

DESCRLONG   = """World Fertily Surveys: Fiji Fertility Survey.
Data represents grouped individual data."""

#suggested notes
NOTE        = """
Number of observations: 70
Number of variables: 7
Variable name definitions:

totchild - total number of children ever born in the group
dur - marriage duration (1=0-4, 2=5-9, 3=10-14, 4=15-19, 5=20-24, 6=25-29)
res - residence (1=Suva, 2=Urban, 3=Rural)
edu - education (1=none, 2=lower primary, 3=upper primary, 4=secondary+)
nwomen - number of women in the group
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath
from scikits.statsmodels.tools import categorical

def load():
    """
    Load the Fiji WFS data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    data = recfromtxt(filepath + '/wfs.csv', delimiter=",",
            names=True, dtype=float, usecols=(1,2,3,4,6))
    names = ["totchild"] +  list(data.dtype.names)
##### SET THE INDEX #####
    endog = array(data[names[4]]*data[names[5]], dtype=float)
    endog_name = names[0]
##### SET THE INDEX #####
    exog = column_stack(data[i] for i in names[1:4]+[names[5]]).astype(float)
    exog_name = names[1:4] + [names[5]]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
