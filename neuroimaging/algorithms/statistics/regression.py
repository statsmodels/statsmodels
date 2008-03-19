"""
This module provides various regression analysis techniques to model the
relationship between the dependent and independent variables.
"""

__docformat__ = 'restructuredtext'

import os

import numpy as np

from neuroimaging.core.api import Image, save_image, slice_iterator

class LinearModelIterator(object):
    """
    TODO
    """

    def __init__(self, iterator, outputs=()):
        """
        :Parameters:
            iterator : TODO
                TODO
            outputs : TODO
                TODO
        """
        self.iterator = iter(iterator)
        self.outputs = [iter(output) for output in outputs]


    def model(self):
        """
        This method should take the iterator at its current state and
        return a LinearModel object.

        :Returns: ``None``
        """
        raise NotImplementedError

    def fit(self):
        """
        Go through an iterator, instantiating model and passing data,
        going through outputs.

        :Returns: ``None``
        """

        for data in self.iterator:
            shape = data.shape[1:]
            data.shape = (data.shape[0], np.product(shape))

            results = self.model().fit(data)
            for output in self.outputs:
                out = output.extract(results)
                if output.nout > 1:
                    out.shape = (output.nout,) + shape
                else:
                    out.shape = shape

                output.set_next(data=out)


class RegressionOutput(object):

    """
    A generic output for regression. Key feature is that it has
    an \'extract\' method which is called on an instance of
    Results.
    """

    def __init__(self, grid, nout=1, outgrid=None):
        """
        :Parameters:
            grid : TODO
                TODO
            nout : ``int``
                TODO
            outgrid : TODO
                TODO
        """
        self.grid = grid
        self.nout = nout
        if outgrid is not None:
            self.outgrid = outgrid
        else:
            self.outgrid = grid
        self.img = NotImplemented
        self.it = NotImplemented

    def __iter__(self):
        """
        :Returns: ``self``
        """
        iter(self.it)
        return self

    def next(self):
        """
        :Returns: TODO
        """
        return self.it.next()

    def set_next(self, data):
        """
        :Parameters:
            data : TODO
                TODO

        :Returns: ``None``
        """
        self.it.next().set(data)

    def extract(self, results):
        """
        :Parameters:
            results : TODO
                TODO

        :Returns: ``None``

        :Raises: NotImplementedError
        """
        raise NotImplementedError

    def _setup_img(self, clobber, outdir, ext, basename):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outname = os.path.join(outdir, '%s%s' % (basename, ext))
        empty = Image(np.zeros(self.outgrid.shape), self.outgrid)
        save_image(outname, clobber=clobber)
        if self.it is NotImplemented:
            it = slice_iterator(img, mode='w')
        else:
            it = iter(self.it.copy(img))
        return img, it

class ImageRegressionOutput(RegressionOutput):
    """
    A class to output things in GLM passes through Image data. It
    uses the image's iterator values to output to an image.
    """

    def __init__(self, grid, nout=1, outgrid=None, it=None):
        """
        :Parameters:
            grid : TODO
                TODO
            nout : int
                TODO
            outgrid : TODO
                TODO
            it : TODO
                TODO
        """
        RegressionOutput.__init__(self, grid, nout, outgrid)

        if self.nout > 1:
            self.grid = self.grid.replicate(self.nout)

        self.img = Image(N.zeros(outgrid.shape), grid=outgrid)
        if it is None:
            self.it = self.img.slice_iterator(mode='w')
        else:
            self.it = it


class TContrastOutput(ImageRegressionOutput):
    """
    TODO
    """

    def __init__(self, grid, contrast, path='.', subpath='contrasts', ext=".nii",
                 effect=True, sd=True, t=True, nout=1, outgrid=None,
                 clobber=False):
        """
        :Parameters:
            grid : TODO
                TODO
            contrast : TODO
                TODO
            path : ``string``
                TODO
            subpath : ``string``
                TODO
            ext : ``string``
                TODO
            effect : ``bool``
                TODO
            sd : ``bool``
                TODO
            t : ``bool``
                TODO
            nout : ``int``
                TODO
            outgrid : TODO
                TODO
            clobber : ``bool``
                TODO
        """
        ImageRegressionOutput.__init__(self, grid, nout, outgrid)
        self.contrast = contrast
        self.effect = effect
        self.sd = sd
        self.t = t
        self._setup_contrast()
        self._setup_output(clobber, path, subpath, ext)

    def _setup_contrast(self, **extra):
        self.contrast.getmatrix(**extra)

    def _setup_output(self, clobber, path, subpath, ext):
        outdir = os.path.join(path, subpath, self.contrast.name)
        self.timg, self.timg_it = self._setup_img(clobber, outdir, ext, 't')

        if self.effect:
            self.effectimg, self.effectimg_it = self._setup_img(clobber, outdir, ext, 'effect')
        if self.sd:
            self.sdimg, self.sdimg_it = self._setup_img(clobber, outdir, ext, 'sd')

        outname = os.path.join(outdir, 'matrix.csv')
        outfile = file(outname, 'w')
        outfile.write(','.join(fpformat.fix(x,4) for x in self.contrast.matrix) + '\n')
        outfile.close()

        outname = os.path.join(outdir, 'matrix.bin')
        outfile = file(outname, 'w')
        self.contrast.matrix = self.contrast.matrix.astype('<f8')
        self.contrast.matrix.tofile(outfile)
        outfile.close()

    def extract(self, results):
        """
        :Parameters:
            results : TODO
                TODO

        :Returns: TODO
        """
        return results.Tcontrast(self.contrast.matrix, sd=self.sd, t=self.t)

    def set_next(self, data):
        """
        :Parameters:
            data : TODO
                TODO

        :Returns: ``None``
        """
        self.timg_it.next().set(data.t)
        if self.effect:
            self.effectimg_it.next().set(data.effect)
        if self.sd:
            self.sdimg_it.next().set(data.sd)


class FContrastOutput(ImageRegressionOutput):
    """
    TODO
    """

    def __init__(self, grid, contrast, path='.', clobber=False,
                 subpath='contrasts', ext='.nii', nout=1, outgrid=None):
        """
        :Parameters:
            grid : TODO
                TODO
            contrast : TODO
                TODO
            path : ``string``
                TODO
            clobber : ``bool``
                TODO
            subpath : ``string``
                TODO
            ext : ``string``
                TODO
            nout : ``int``
                TODO
            outgrid : TODO
                TODO
        """
        ImageRegressionOutput.__init__(self, grid, nout, outgrid)
        self.contrast = contrast
        self._setup_contrast()
        self._setup_output(clobber, path, subpath, ext)

    def _setup_contrast(self, **extra):
        self.contrast.getmatrix(**extra)

    def _setup_output(self, clobber, path, subpath, ext):
        outdir = os.path.join(path, subpath, self.contrast.name)
        self.img, self.it = self._setup_img(clobber, outdir, ext, 'F')

        outname = os.path.join(outdir, 'matrix.csv')
        outfile = file(outname, 'w')
        writer = csv.writer(outfile)
        for row in self.contrast.matrix:
            writer.writerow([fpformat.fix(x, 4) for x in row])
        outfile.close()

        outname = os.path.join(outdir, 'matrix.bin')
        outfile = file(outname, 'w')
        self.contrast.matrix = self.contrast.matrix.astype('<f8')
        self.contrast.matrix.tofile(outfile)
        outfile.close()

    def extract(self, results):
        """
        :Parameters:
            results : TODO
                TODO
        """
        return results.Fcontrast(self.contrast.matrix).F


class ResidOutput(ImageRegressionOutput):
    """
    TODO
    """

    def __init__(self, grid, path='.', nout=1, clobber=False, basename='resid',
                 ext='.nii', outgrid=None):
        """
        :Parameters:
            grid : TODO
                TODO
            path : ``string``
                TODO
            nout : ``int``
                TODO
            clobber : ``bool``
                TODO
            basename : ``string``
                TODO
            ext : ``string``
                TODO
            outgrid : TODO
                TODO
        """
        ImageRegressionOutput.__init__(self, grid, nout, outgrid)
        outdir = os.path.join(path)

        self.img, self.it = self._setup_img(clobber, outdir, ext, basename)
        self.nout = self.grid.shape[0]

    def extract(self, results):
        """
        :Parameters:
            results : TODO
                TODO

        :Returns: TODO
        """
        return results.resid
