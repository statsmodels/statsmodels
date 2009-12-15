"""
This should be merged into statsmodels/tests/model_results.py when things
move out of the sandbox.
"""
import numpy as np

class Anes(object):
    def __init__(self):
        """
        Results are from R nnet package.
        """
        params = np.array([-0.373356261, -2.250934805, -3.665905084,
            -7.613694423, -7.060431370, -12.105193452, -0.011537359,
            -0.088750964, -0.105967684, -0.091555188, -0.093285749,
            -0.140879420,  0.297697981,  0.391662761,  0.573513420,
            1.278742543,  1.346939966,  2.069988287, -0.024944529,
            -0.022897526, -0.014851243, -0.008680754, -0.017903442,
            -0.009432601,  0.082487696, 0.181044184, -0.007131611,
            0.199828063,  0.216938699,  0.321923127,  0.005195818,
            0.047874118,  0.057577321,  0.084495215,  0.080958623, 0.108890412])
#TODO: the below will change when we return a different
        self.params = params.reshape(6,-1).T

