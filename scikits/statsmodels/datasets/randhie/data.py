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

"""Name of dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is in the public domain."""
TITLE       = ""
SOURCE      = """
The data was collected by the RAND corporation as part of the Health
Insurance Experiment (HIE).

http://www.rand.org/health/projects/hie/

This data was used in

Cameron, A.C. amd Trivedi, P.K. 2005.  `Microeconometrics: Methods
    and Applications,` Cambridge: New York.

And was obtained from: <http://cameron.econ.ucdavis.edu/mmabook/mmadata.html>

See randhie/src for the original data and description.  The data included
here contains only a subset of the original data.  The data varies slightly
compared to that reported in Cameron and Trivedi.
"""

DESCRSHORT  = """The RAND Co. Health Insurance Experiment Data"""

DESCRLONG   = """"""

NOTE        = """
Number of observations: 20,190

Variables
----------
mdvis - Number of outpatient visits to an MD
lncoins - ln(coinsurance + 1), 0 <= coninsurance <= 100
idp - 1 if individual deductible plan, 0 otherwise
lpi - ln(max(1, annual participation incentive payment))
fmde - 0 if idp = 1; ln(max(1, MDE/(0.01 coinsurance))) otherwise
physlm - 1 if the person has a physical limitation
disea - number of chronic diseases
hlthg - 1 if self-rated health is good
hlthf - 1 if self-rated health is fair
hlthp - 1 if self-rated health is poor
        (Omitted category is excellent self-rated health)
"""

import numpy as np

class Load():
    """Loads the RAND HIE data and returns a data class.

    Attributes
    ----------
    endog - structured array of response variable, mdvis
    exog - strucutured array of design

    Returns
    Load instance:
        a class of the data with array attrbutes 'endog' and 'exog'
    """
    def __init__(self):
        from randhie import __dict__, names
        self._names = names
        self._d = __dict__
#        nobs = len(__dict__[names[0]])
#        endog = np.zeros(nobs, dtype=[('mdvis', float)])
        self.endog = np.array(__dict__[names[0]]).astype(float)
        design_dt = [_.lower() for _ in names]
        design_dt = np.dtype(zip(design_dt, [float]*len(design_dt)))
        exog = np.zeros(len(self.endog), dtype=design_dt)
        for i in design_dt.names[1:]:
            exog[i] = np.array(__dict__[i.upper()]).astype(float)
        self.exog = exog
