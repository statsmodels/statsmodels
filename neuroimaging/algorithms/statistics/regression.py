"""
This module provides various convenience functions for extracting
statistics from regression analysis techniques to model the
relationship between the dependent and independent variables.
"""

__docformat__ = 'restructuredtext'

import os

import numpy as np

from neuroimaging.core.api import Image, save_image

def output_T(results, contrast, effect=None, sd=None, t=None):
    """
    This convenience function outputs the results of a Tcontrast
    from a regression
    """
    if not hasattr(self, "contrast"):
        contrast.getmatrix()
    r = results.Tcontrast(self.contrast.matrix, sd=sd,
                          t=t)
    # this may not always be an array..
    return [r.effect,
            r.sd,
            r.t]

    r = results.Tcontrast(contrast.matrix)

def output_F(results, contrast):
    """
    This convenience function outputs the results of an Fcontrast
    from a regression
    """
    if not hasattr(self, "contrast"):
        contrast.getmatrix()
    return results.Fcontrast(contrast.matrix).F

def output_resid(results):
    """
    This convenience function outputs the residuals
    from a regression
    """
    return results.resid

