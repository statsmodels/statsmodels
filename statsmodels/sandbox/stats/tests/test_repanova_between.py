from __future__ import print_function

# Copyright (c) 2013, Roger Lew [see LICENSE.txt]

from StringIO import StringIO

import unittest
import pandas
import numpy as np

from statsmodels.sandbox.stats import repanova as anova

data_tbl = StringIO("""\
SUBJECT,AGE,CONDITION,WORDS,X
1,old,counting,9,81
2,old,counting,8,64
3,old,counting,6,36
4,old,counting,8,64
5,old,counting,10,100
6,old,counting,4,16
7,old,counting,6,36
8,old,counting,5,25
9,old,counting,7,49
10,old,counting,7,49
11,old,rhyming,7,49
12,old,rhyming,9,81
13,old,rhyming,6,36
14,old,rhyming,6,36
15,old,rhyming,6,36
16,old,rhyming,11,121
17,old,rhyming,6,36
18,old,rhyming,3,9
19,old,rhyming,8,64
20,old,rhyming,7,49
21,old,adjective,11,121
22,old,adjective,13,169
23,old,adjective,8,64
24,old,adjective,6,36
25,old,adjective,14,196
26,old,adjective,11,121
27,old,adjective,13,169
28,old,adjective,13,169
29,old,adjective,10,100
30,old,adjective,11,121
31,old,imagery,12,144
32,old,imagery,11,121
33,old,imagery,16,256
34,old,imagery,11,121
35,old,imagery,9,81
36,old,imagery,23,529
37,old,imagery,12,144
38,old,imagery,10,100
39,old,imagery,19,361
40,old,imagery,11,121
41,old,intention,10,100
42,old,intention,19,361
43,old,intention,14,196
44,old,intention,5,25
45,old,intention,10,100
46,old,intention,11,121
47,old,intention,14,196
48,old,intention,15,225
49,old,intention,11,121
50,old,intention,11,121
51,young,counting,8,64
52,young,counting,6,36
53,young,counting,4,16
54,young,counting,6,36
55,young,counting,7,49
56,young,counting,6,36
57,young,counting,5,25
58,young,counting,7,49
59,young,counting,9,81
60,young,counting,7,49
61,young,rhyming,10,100
62,young,rhyming,7,49
63,young,rhyming,8,64
64,young,rhyming,10,100
65,young,rhyming,4,16
66,young,rhyming,7,49
67,young,rhyming,10,100
68,young,rhyming,6,36
69,young,rhyming,7,49
70,young,rhyming,7,49
71,young,adjective,14,196
72,young,adjective,11,121
73,young,adjective,18,324
74,young,adjective,14,196
75,young,adjective,13,169
76,young,adjective,22,484
77,young,adjective,17,289
78,young,adjective,16,256
79,young,adjective,12,144
80,young,adjective,11,121
81,young,imagery,20,400
82,young,imagery,16,256
83,young,imagery,16,256
84,young,imagery,15,225
85,young,imagery,18,324
86,young,imagery,16,256
87,young,imagery,20,400
88,young,imagery,22,484
89,young,imagery,14,196
90,young,imagery,19,361
91,young,intention,21,441
92,young,intention,19,361
93,young,intention,17,289
94,young,intention,15,225
95,young,intention,22,484
96,young,intention,16,256
97,young,intention,22,484
98,young,intention,22,484
99,young,intention,18,324
100,young,intention,21,441""")

class Test_anova_between(unittest.TestCase):
    def test2(self):
        ## Between-Subjects test

        R = """WORDS ~ AGE * CONDITION

TESTS OF BETWEEN-SUBJECTS EFFECTS

Measure: WORDS
    Source        Type III   df     MS        F        Sig.      et2_G   Obs.    SE     95% CI   lambda   Obs.  
                     SS                                                                                   Power 
===============================================================================================================
AGE                240.250    1   240.250   29.936   3.981e-07   0.250     50   0.406    0.796   16.631   0.981 
CONDITION         1514.940    4   378.735   47.191   2.530e-21   0.677     20   0.642    1.258   41.948   1.000 
AGE * CONDITION    190.300    4    47.575    5.928   2.793e-04   0.209     10   0.908    1.780    2.635   0.207 
Error              722.300   90     8.026                                                                       
===============================================================================================================
Total             2667.790   99                                                                                 

TABLES OF ESTIMATED MARGINAL MEANS

Estimated Marginal Means for AGE
 AGE     Mean    Std. Error   95% Lower Bound   95% Upper Bound 
===============================================================
old     10.060        0.567             8.949            11.171 
young   13.160        0.818            11.556            14.764 

Estimated Marginal Means for CONDITION
CONDITION    Mean    Std. Error   95% Lower Bound   95% Upper Bound 
===================================================================
adjective   12.900        0.791            11.350            14.450 
counting     6.750        0.362             6.041             7.459 
imagery     15.500        0.933            13.671            17.329 
intention   15.650        1.096            13.502            17.798 
rhyming      7.250        0.452             6.363             8.137 

Estimated Marginal Means for AGE * CONDITION
 AGE    CONDITION    Mean    Std. Error   95% Lower Bound   95% Upper Bound 
===========================================================================
old     adjective       11        0.789             9.454            12.546 
old     counting         7        0.577             5.868             8.132 
old     imagery     13.400        1.424            10.610            16.190 
old     intention       12        1.183             9.681            14.319 
old     rhyming      6.900        0.674             5.579             8.221 
young   adjective   14.800        1.104            12.637            16.963 
young   counting     6.500        0.453             5.611             7.389 
young   imagery     17.600        0.819            15.994            19.206 
young   intention   19.300        0.844            17.646            20.954 
young   rhyming      7.600        0.618             6.388             8.812 

"""
        data_tbl.seek(0)
        df = pandas.io.parsers.read_csv(data_tbl)
        aov_results = anova.anova(df, 'WORDS',
		                          bfactors=['AGE','CONDITION'], 
								  sub='SUBJECT')                          
        D = aov_results.summary()
        self.assertEqual(R,D)
        
def suite():
    return unittest.TestSuite((
            unittest.makeSuite(Test_anova_between)
                              ))

if __name__ == "__main__":
    # run tests
    runner = unittest.TextTestRunner()
    runner.run(suite())
    
