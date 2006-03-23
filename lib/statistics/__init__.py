import model
import regression
import classification
import iterators

# Is this import now redundant?  The utils referred to seems to now be
# in fmri/utils

# not necessarily -- there are some differences. i suppose most utils
# should be moved to here.. this is the natural place for a lot of them
# i would like to release statistics as a separate package, too
# under a BSD license rather than GPL

import utils

import unittest
def suite():
    return unittest.TestSuite([tests.suite()])

