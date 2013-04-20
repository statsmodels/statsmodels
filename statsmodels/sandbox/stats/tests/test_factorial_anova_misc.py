from __future__ import print_function

# Copyright (c) 2013, Roger Lew [see LICENSE.txt]

from StringIO import StringIO

import unittest
import pandas
import numpy as np

from statsmodels.sandbox.stats import factorial_anova as anova

class Test_noncentrality_parameter(unittest.TestCase):
    def test1(self):
        r = anova.noncentrality_parameter(100,10,89)
        d = 890.
        self.assertAlmostEqual(r, d)
    
Test_noncentrality_parameter
class Test_observed_power(unittest.TestCase):
    def test1(self):
        # from http://zoe.bme.gatech.edu/~bv20/public/samplesize.pdf
        # page 9
        r = anova.observed_power(3, 60, 16)
        self.assertAlmostEqual(r, 0.91672320189642831)
            
    def test2(self):
        # test eps kwd
        r = anova.observed_power(3, 60, 16, eps = .5)
        d = anova.observed_power(1.5, 30, 8)
        self.assertAlmostEqual(r, d)
        
    def test3(self):
        # Gpower 3.1.3
        # Test family = F tests
        # Statistical test = ANOVA: Feixed effects, omnibus, one-way
        # Type of power analysis = Post hoc: Compute achieved power
        #
        # Input Parameters
        # Effect size f = 0.25
        # alpha err prob = 0.01
        # Tot. sample size = 100
        # Num. of groups = 5

        d = 0.2361189 # from Gpower

        r = anova.observed_power(4, 95, 6.25, alpha =.01)
        self.assertAlmostEqual(r, d, 4) # matches to 4 decimal places
        
    def test4(self):
        # Gpower 3.1.3
        # Test family = F tests
        # Statistical test = ANOVA: Feixed effects, omnibus, one-way
        # Type of power analysis = Post hoc: Compute achieved power
        #
        # Input Parameters
        # Effect size f = 0.25
        # alpha err prob = 0.01
        # Tot. sample size = 100
        # Num. of groups = 2

        d = 0.9903248 # from Gpower

        r = anova.observed_power(1, 98, 25, alpha =.01)
        self.assertAlmostEqual(r, d, 6) # matches to 6 decimal places
        
class Test_eps(unittest.TestCase):
    def setUp(self):
        
        x = np.array([[ 450.,  660.,  720.],
                      [ 510.,  720.,  510.],
                      [ 630.,  510.,  660.],
                      [ 390.,  660.,  780.],
                      [ 480.,  630.,  510.],
                      [ 540.,  360.,  660.],
                      [ 570.,  450.,  660.],
                      [ 630.,  450.,  510.],
                      [ 660.,  510.,  540.],
                      [ 450.,  600.,  660.]])

        c = np.array([[ 0.66666667, -0.33333333, -0.33333333],
                      [-0.33333333,  0.66666667, -0.33333333],
                      [-0.33333333, -0.33333333,  0.66666667]])

        self.y = np.dot(x,c)
        self.df = 2

    def test_epsLB(self):
        
        r = anova._epsGG(self.y, self.df)
        self.assertAlmostEqual(r, 0.967621240466)
            
    def test_epsHF(self):
        
        r = anova._epsHF(self.y, self.df)
        self.assertAlmostEqual(r, 0.967621240466)
        
    def test_epsLB(self):
        
        r = anova._epsLB(self.y, self.df)
        self.assertAlmostEqual(r, 0.5)


class Test_windsortrim(unittest.TestCase):
    def setUp(self):
        self.x = np.array([ 3,  7, 12, 15, 17, 17, 18, 19, 19, 19,
                           20, 22, 24, 26, 30, 32, 32, 33, 36, 50])
        
    def test1(self):
        d = np.array([12, 12, 12, 15, 17, 17, 18, 19, 19, 19,
                      20, 22, 24, 26, 30, 32, 32, 33, 33, 33])

        r,numtrimmed = anova.windsor(self.x, .10)

        np.testing.assert_array_almost_equal(r, d)
        self.assertEqual(numtrimmed, 4)
        
    def test2(self):
        d = np.array([17, 17, 17, 17, 17, 17, 18, 19, 19, 19,
                      20, 22, 24, 26, 30, 32, 32, 32, 32, 32])

        r,numtrimmed = anova.windsor(self.x, .2)

        np.testing.assert_array_almost_equal(r, d)
        self.assertEqual(numtrimmed, 8)
        
        
def suite():
    return unittest.TestSuite((
            unittest.makeSuite(Test_noncentrality_parameter),
            unittest.makeSuite(Test_observed_power),
            unittest.makeSuite(Test_eps),
            unittest.makeSuite(Test_windsortrim) 
                              ))

if __name__ == "__main__":
    # run tests
    runner = unittest.TextTestRunner()
    runner.run(suite())
    
