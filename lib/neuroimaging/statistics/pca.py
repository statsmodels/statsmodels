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
from enthought import traits

from neuroimaging.image import Image
from neuroimaging.statistics.utils import recipr

class PCA(traits.HasTraits):
    """
    Compute the PCA of an image (over axis=0). Image grid should
    have a subgrid method.
    """

    design_keep = traits.Array(shape=(None,None), desc='Data is projected onto the column span of design_keep.')
    design_resid = traits.Array(shape=(None,None), desc='After projecting onto the column span of design_keep, data is projected off of the column span of this matrix.')
    tol = traits.Float(1.0e-05)
    pcatype = traits.Trait('cor','cov')
    mask = traits.Instance(Image)
    ext = traits.String('.img')

    def __init__(self, image, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.image = image

        if self.mask is not None:
            self._mask = N.array(self.mask.readall())
            self.nvoxel = self._mask.sum()
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

        if N.allclose(self.design_keep, [[0]]):
            self.design_resid = N.ones((self.nimages, 1), N.Float)

        if N.allclose(self.design_keep, [[0]]):
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
                mask = self._mask[i]
                mask.shape = N.product(mask.shape)
                YX = YX * mask

            self.C += N.dot(YX, N.transpose(YX))

        self.D, self.Vs = L.eigh(self.C)
        order = N.argsort(-self.D)
        self.D = self.D[order]
        self.pcntvar = self.D * 100 / self.D.sum()

        self.components = N.transpose(N.dot(N.transpose(UX), self.Vs))[order]

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
                mask = self._mask[i]
                mask.shape = N.product(mask.shape)
                U = U * mask

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

try:
    import pylab
    from neuroimaging.visualization.montage import Montage
    from neuroimaging.image.interpolation import ImageInterpolator
    from neuroimaging.visualization import slices
    from neuroimaging.fmri.plotting import MultiPlot

    class PCAmontage(PCA):

        """
        Same as PCA but with a montage method to view the resulting images
        and a time_series image to view the time components.

        Note that the results of calling images are stored for this class,
        therefore to free the memory of the output of images, the
        image_results attribute of this instance will also have to be deleted.
        """

        image_results = traits.Any()

        def images(self, which=[0], output_base=None):
            PCA.images.__doc__
            self.image_results = PCA.images(self, which=which, output_base=output_base)
            self.image_which = which
            return self.image_results

        def time_series(self, title='Principal components in time'):
            """
            Plot the time components from the last call to 'images' method.
            """

            pylab.clf()

            if self.image_results is None:
                raise ValueError, 'run "images" before time_series'
            try:
                t = self.image.frametimes
            except:
                t = N.arange(self.image.grid.shape[0])
            self.time_plot = MultiPlot(self.components[self.image_which],
                                       time=t,
                                       title=title)
            self.time_plot.draw()

        def montage(self, z=None, nslice=None, xlim=[-120,120], ylim=[-120,120],
                    colormap='spectral', width=10):
            """
            Plot a montage of transversal slices from last call to
            'images' method.

            If z is not specified, a range of nslice equally spaced slices
            along the range of the first axis of image_results[0].grid is used,
            where nslice defaults to image_results[0].grid.shape[0].

            """

            pylab.clf()

            if nslice is None:
                nslice = self.image_results[0].grid.shape[0]
            if self.image_results is None:
                raise ValueError, 'run "images" before montage'
            images = self.image_results
            nrow = len(images)

            if z is None:
                r = images[0].grid.range()
                zmin = r[0].min(); zmax = r[0].max()
                z = N.linspace(zmin, zmax, nslice)

            z = list(N.asarray(z).flat)
            z.sort()
            ncol = len(z)

            basegrid = images[0].grid
            if self.mask is not None:
                mask_interp = ImageInterpolator(mask)
            else:
                mask_interp = None

            montage_slices = {}

            image_interps = [ImageInterpolator(images[i]) for i in range(nrow)]
            interp_slices = [slices.transversal(basegrid,
                                                z=zval,
                                                xlim=xlim,
                                                ylim=ylim) for zval in z]

            vmax = N.array([images[i].readall().max() for i in range(nrow)]).max()
            vmin = N.array([images[i].readall().min() for i in range(nrow)]).min()

            for i in range(nrow):

                for j in range(ncol):

                    montage_slices[(nrow-1-i,ncol-1-j)] = \
                       slices.DataSlicePlot(image_interps[i],
                                            interp_slices[j],
                                            vmax=vmax,
                                            vmin=vmin,
                                            colormap=colormap,
                                            interpolation='nearest',
                                            mask=mask_interp,
                                            transpose=True)

            m = Montage(slices=montage_slices, vmax=vmax, vmin=vmin)
            m.draw()

    class MultiPlot(traits.HasTraits):
        """
        Class to plot multi-valued time series simultaneously.

        Should be moved somewhere better.
        """

        figure = traits.Any()
        title = traits.Str()
        time = traits.Array(shape=(None,))

        def __init__(self, series, **keywords):
            self.series = series
            traits.HasTraits.__init__(self, **keywords)
            self.figure = pylab.gcf()

        def draw(self, **keywords):
            pylab.figure(num=self.figure.number)

            self.lines = []

            v = self.series
            if v.ndim == 1:
                v.shape = (1, v.shape[0])
            v = v[::-1]

            if N.allclose(self.time, N.array([0])):
                self.time = N.arange(v.shape[1])
            n = v.shape[0]
            dy = 0.9 / n
            for i in range(n):
                a = pylab.axes([0.05,0.05+i*dy,0.9,dy])
                a.set_xticklabels([])
                a.set_yticks([])
                a.set_yticklabels([])
                m = N.nanmin(v[i])
                M = N.nanmax(v[i])
                pylab.plot(self.time, v[i])
                r = M - m
                l = m - 0.2 * r
                u = M + 0.2 * r
                if l == u:
                    u += 1.
                    l -= 1.
                a.set_ylim([l, u])

            pylab.title(self.title)


except:
    pass

