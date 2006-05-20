import sets, time, string, gc

import numpy as N
import numpy.linalg as L
#import numpy.random as R

from neuroimaging.image import Image
from utils import recipr
import enthought.traits as traits

class PCA(traits.HasTraits):
    """
    Compute the PCA of an image (over axis=0). Image grid should
    have a subgrid method.
    """

    design_resid = traits.Any()
    design_keep = traits.Any()
    tol = traits.Float(1.0e-05)
    pcatype = traits.Trait('cor','cov')
    mask = traits.Any()
    ext = traits.String('.img')

    def __init__(self, image, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.image = image

        if self.mask is not None:
            self.mask = self.mask.readall()
            self.nvoxel = self.mask.sum()
        else:
            self.nvoxel = N.product(self.image.grid.shape[1:])

        self.nimages = self.image.grid.shape[0]

    def _design_keep_changed(self):
        self.proj_keep = N.dot(self.design_keep, L.pinv(self.design_keep))

    def _design_resid_changed(self):
        self.proj_resid = N.dot(self.design_resid, L.pinv(self.design_resid))

    def project(self, Y, which='keep'):
        if which == 'keep':
            if self.design_keep is None:
                return Y
            else:
                return N.dot(self.proj_keep, Y)
        else:
            if self.design_resid is None:
                return Y
            else:
                return Y - N.dot(self.proj_resid, Y)

    def fit(self):
        """
        Perform the computations needed for the PCA.
        """
        # Compute projection matrices

        if self.design_resid is None: # always remove the mean
            self.design_resid = N.ones((self.nimages, 1), N.Float)

        if self.design_keep is None:
            self.design_keep = N.identity(self.nimages)

        X = N.dot(self.design_keep, L.pinv(self.design_keep))
        XZ = X - N.dot(self.design_resid, N.dot(L.pinv(self.design_resid), X))
        UX, SX, VX = L.svd(XZ, full_matrices=0)

        rank = N.greater(SX/SX.max(), 0.5).astype(N.Int).sum()
        UX = N.transpose(UX[:,range(rank)])

        first_slice = slice(0,self.image.shape[0])
        _shape = self.image.grid.shape
        A = N.zeros((rank,)*2, N.Float)

        for i in range(self.image.shape[1]):
            _slice = [first_slice, slice(i,i+1)]
            Y = self.image.getslice(_slice).reshape((_shape[0], N.product(_shape[2:])))
            YX = N.dot(UX, Y)

            if self.pcatype == 'cor':
                S2 = N.add.reduce(self.project(Y, which='resid')**2, axis=0)
                Smhalf = recipr(N.sqrt(S2))
                del(S2); gc.collect()
                YX = YX * Smhalf

            if self.mask is not None:
                YX = YX * self.mask

            A += N.dot(YX, N.transpose(YX))

        self.D, self.Vs = L.eigh(A)
        order = N.argsort(-self.D)
        self.D = self.D[order]
        self.pcntvar = self.D * 100 / self.D.sum()

        self.VX = N.transpose(N.dot(N.transpose(UX), N.transpose(self.Vs[:,order])))

    def components(self, which=[0], output_base=None):
        """
        Output the component images -- by default, only output the first
        principal component.
        """

        ncomp = len(which)
        subVX = self.VX[which]

        outgrid = self.image.grid.subgrid(0)
        if output_base is not None:
            outimages = [Image('%s_comp%d.%d' % (output_base, i, self.ext),
                               grid=outgrid)
                        for i in which]
        else:
            outimages = [Image(N.zeros(outgrid.shape, N.Float), grid=outgrid)]

        first_slice = slice(0,self.image.shape[0])
        _shape = self.image.grid.shape

        for i in range(self.image.shape[1]):
            _slice = [first_slice, slice(i,i+1)]
            Y = self.image.getslice(_slice).reshape((_shape[0], N.product(_shape[2:])))
            U = N.dot(subVX, Y)

            if self.mask is not None:
                U = U * self.mask

            if self.pcatype == 'cor':
                S2 = N.add.reduce(self.project(Y, which='resid')**2, axis=0)
                Smhalf = recipr(N.sqrt(S2))
                del(S2); gc.collect()
                U = U * Smhalf

            del(Y); gc.collect()

            U.shape = (U.shape[0],) + outgrid.shape
            for k in range(len(which)):
                outimages[k].next(data=U[k])

            del(U); gc.collect()

        return outimages


## def normalize_max(image, nvector=None, **keywords):
##     input_image = BrainSTAT.VImage(image, **keywords)
##     input_data = input_image.toarray(warp=input_image.warp).image.data

##     if nvector is None:
##         curmin = min(input_data.flat)
##         curmax = max(input_data.flat)
##         ok = (curmax > -curmin)
##         absmax = max(abs(curmax), abs(curmin))
##         if ok:
##             input_data = input_data / absmax
##         else:
##             input_data = -input_data / absmax
##         output_image = BrainSTAT.VImage(input_data, warp=input_image.warp)
##         return output_image

##     else:
##         for i in range(nvector):
##             curmin = min(input_data[:,:,:,i].flat)
##             curmax = max(input_data[:,:,:,i].flat)
##             ok = (curmax > -curmin)
##             absmax = max(abs(curmax), abs(curmin))
##             if ok:
##                 input_data[:,:,:,i] = input_data[:,:,:,i] / absmax
##             else:
##                 input_data[:,:,:,i] = -input_data[:,:,:,i] / absmax
##         output_image = BrainSTAT.VImage(input_data, warp=input_image.warp)
##     return output_image

if __name__ == "__main__":

    from neuroimaging.fmri import fMRIImage
    image = fMRIImage('http://kff.stanford.edu/BrainSTAT/testdata/test_fmri.img')
    p = PCA(image)
    p.fit()
    a = p.components(which=[0,2,3])
