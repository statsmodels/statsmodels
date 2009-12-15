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

"""American National Election Survey 1996"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = ""
SOURCE      = """
http://www.electionstudies.org/

The American National Election Studies.
"""

DESCRSHORT  = """This data is a subset of the American National Election Studies of 1996."""

DESCRLONG   = DESCRSHORT

NOTE        = """
Number of observations: 944

This is just an example dataset and there is no guarantee that the below
is wholly accurate.

Variables
----------
popul - Census place population in 1000s

TVnews - Number of times per week that respondent watches TV news.

PID - Polytomous variable. Party identification of respondent.
    0 - Strong Democrat
    1 - Weak Democrat
    2 - Independent-Democrat
    3 - Independent-Indpendent
    4 - Independent-Republican
    5 - Weak Republican
    6 - Strong Republican

age - Age of respondent.

educ - Polytomous variable.
    1 - 1-8 grades
    2 - Some high school
    3 - High school graduate
    4 - Some college
    5 - College degree
    6 - Master's degree
    7 - PhD

income - Polytmous variable. Income of household
    1 - None or less than $2,999
    2 - $3,000-$4,999
    3 - $5,000-$6,999
    4 - $7,000-$8,999
    5 - $9,000-$9,999
    6 - $10,000-$10,999
    7 - $11,000-$11,999
    8 - $12,000-$12,999
    9 - $13,000-$13,999
    10 - $14,000-$14.999
    11 - $15,000-$16,999
    12 - $17,000-$19,999
    13 - $20,000-$21,999
    14 - $22,000-$24,999
    15 - $25,000-$29,999
    16 - $30,000-$34,999
    17 - $35,000-$39,999
    18 - $40,000-$44,999
    19 - $45,000-$49,999
    20 - $50,000-$59,999
    21 - $60,000-$74,999
    22 - $75,000-89,999
    23 - $90,000-$104,999
    24 - $105,000 and over

vote - Whether or not the respondent voted in the previous presidential
        election.
    0 - No
    1 - Yes

The following 3 variables all take the values:
    1 - Extremely liberal
    2 - Liberal
    3 - Slightly liberal
    4 - Moderate
    5 - Slightly conservative
    6 - Conservative
    7 - Extremely Conservative
selfLR - Polytomous variable. Respondent's self-reported political leanings
            from "Left" to "Right".
ClinLR - Polytomous variable. Respondents impression of Bill Clinton's
            political leanings from "Left" to "Right".
DoleLR  - Polytmous variable. Respondents impression of Bob Dole's
            political leanings from "Left" to "Right".
"""

import numpy as np

class Load():
    """Load the anes96 data and returns a data class.

    Returns
    Load instance:
        a class of the data with array attrbutes 'endog' and 'exog'
    """
    def __init__(self):
        from anes96 import __dict__, names
        self._names = names
        self._d = __dict__
        self.endog = np.array(self._d[self._names[5]], dtype=np.float)
        self.exog = np.column_stack(self._d[i] \
                for i in self._names[0:5]+self._names[6:]).astype(np.float)
