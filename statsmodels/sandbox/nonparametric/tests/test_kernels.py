
import numpy.testing as npt
import numpy as np
import statsmodels.sandbox.nonparametric.kernels as kernels


class test_norm_constant():
    
    def test_norm_constant_calculation(self):
        custom_gauss = kernels.CustomKernel(lambda x: np.exp(-x**2/2.0))
        gauss_true_const = 0.3989422804014327
        npt.assert_almost_equal(gauss_true_const, custom_gauss.norm_const)


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
