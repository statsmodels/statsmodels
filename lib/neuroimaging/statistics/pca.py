"""
This module provides a class for principal components analysis (PCA).

PCA is an orthonormal, linear transform (i.e., a rotation) that maps the
data to a new coordinate system such that the  maximal variability of the
data lies on the first coordinate (or the first principal component), the
second greatest variability is projected onto the second coordinate, and
so on.  The resulting data has unit covariance (i.e., it is decorrelated).
This technique can be used to reduce the dimensionality of the data.

More specifically, the data is projected onto the eigenvectors of the
covariance matrix.
"""

import time, gc

import numpy as N
import numpy.linalg as L
#import numpy.random as R
from enthought import traits

from neuroimaging.image import Image
from neuroimaging.statistics.utils import recipr

class PCA(traits.HasTraits):
    """
    Compute the PCA of an image (over axis=0). Image grid should
    have a subgrid method.
    """

    design_resid = traits.Any()
    design_keep = traits.Any()
    tol = traits.Float(1.0e-05)
    pcatype = traits.Trait('cov','cor')
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
        This stores the covariance/correlation matrix of the data in
        the attribute 'C'.
        The components are stored as the attributes 'components',
        for an fMRI image these are the time series explaining the most
        variance.

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
        self.C = N.zeros((rank,)*2, N.Float)

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

            self.C += N.dot(YX, N.transpose(YX))

        self.D, self.Vs = L.eigh(self.C)
        order = N.argsort(-self.D)
        self.D = self.D[order]
        self.pcntvar = self.D * 100 / self.D.sum()

        self.components = N.transpose(N.dot(N.transpose(UX), N.transpose(self.Vs[:,order])))

    def images(self, which=[0], output_base=None):
        """
        Output the component images -- by default, only output the first
        principal component.
        """

        ncomp = len(which)
        subVX = self.components[which]

        outgrid = iter(self.image.grid.subgrid(0))

        if output_base is not None:
            outimages = [iter(Image('%s_comp%d%s' % (output_base, i, self.ext),
                                    grid=outgrid, mode='w')) for i in which]
        else:
            outimages = [iter(Image(N.zeros(outgrid.shape, N.Float),
                                    grid=outgrid)) for i in which]

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

            U.shape = (U.shape[0],) + outgrid.shape[1:]
            itervalue = outgrid.next()
            for k in range(len(which)):
                outimages[k].next(data=U[k], value=itervalue)

        for i in range(len(which)):
            if output_base:
                outimage = iter(Image('%s_comp%d%s' % (output_base, which[i], self.ext),
                                      grid=outgrid, mode='r+'))
            else:
                outimage = outimages[i]
            d = outimage.readall()
            dabs = N.fabs(d); di = dabs.argmax()
            d = d / d.flat[di]
            outslice = [slice(0,j) for j in outgrid.shape]
            outimage.writeslice(outslice, d)
            del(d); gc.collect()

        return outimages

if __name__ == "__main__":

    from neuroimaging.fmri import fMRIImage
    image = fMRIImage('http://kff.stanford.edu/BrainSTAT/testdata/test_fmri.img')
    p = PCA(image)
    p.fit()
    a = p.images(which=range(4))

    from neuroimaging.image.interpolation import ImageInterpolator
    interpolator = ImageInterpolator(a[2])

    r = a[0].grid.range()
    z = N.unique(r[0].flat); y = N.unique(r[1].flat); z = N.unique(r[2].flat)

    from neuroimaging.visualization import slices
    _slices = {}
    vmax = a[0].readall().max(); vmin = a[0].readall().min()
    for i in range(5):
        for j in range(6):

            _slice = slices.transversal(a[2], z=z[i],
                                        xlim=[-150,150.],
                                        ylim=[-150.,150.])
            _slices[i,j] = slices.DataSlicePlot(interpolator, _slice,
                                                vmax=vmax, vmin=vmin,
                                                colormap='spectral',
                                                interpolation='nearest')

    from neuroimaging.visualization.montage import Montage
    m = Montage(slices=_slices, vmax=vmax, vmin=vmin)
    m.draw()
##

##     x =

##     x.width = 0.8; x.height = 0.8
##     pylab.figure(figsize=(5,5))
##     x.getaxes()
##     pylab.imshow(x.RGBA(), origin=x.origin)
    import pylab
    pylab.show()

