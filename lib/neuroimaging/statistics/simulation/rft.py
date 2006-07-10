"""
This module has a collection of functions for the simulation of random fields.
Can be used for numerical validation of RFT.
"""

import numpy as N

import gc

from neuroimaging.image.fwhm import fastFWHM
#from BrainSTAT.Simulation import Simulator
#from BrainSTAT import *
#from BrainSTAT.Modules.UGRF import UGRF
#from BrainSTAT.RFT.EC import *

from neuroimaging import traits

from neuroimaging.statistics.simulation import ugrf, Simulator

class GaussianField(Simulator):

    """
    A class to simulate smooth Gaussian fields based on a UGRF.

    Note: fwhm argument is only used when image is a VImage instance, and is ignored if image is a UGRF instance.
    """

    estimate_resels = traits.false(desc='Produce a resel estimate?')
    ugrf = traits.Instance(ugrf.UGRF, desc='Simulator.')

    def __init__(self, grid, search=None, verbose=False, **keywords):
        self.ugrf = ugrf.UGRF(grid, **keywords)
        Simulator.__init__(self, search=search, verbose=verbose)

        if resels is None and self.estimate_resels:
            self.resels = self.resel_estimator(**keywords)
        else:
            self.resels = None
        self.ec_density = ec_densities.Gaussian
        self._get_pvalue()

    def feature(self, image, **keywords):
        """
        Default feature: return maximum over search region.
        To change, subclass and write over this method.
        """
        data = N.compress(self.search.toVImage(image.warp).readall(**keywords), image.readall(**keywords)).flat
        return N.maximum.reduce(data)

    def _get_pvalue(self, **keywords):
        """
        Default pvalue: if resels present, return the EC approximation
        based on which random field model is being used.
        To change, subclass and write over this method.
        """

        if self.resels:
            volume = self.resels.integrate(self.search)
            search = volume2ball(volume, d=self.ugrf.ndim)
        else:
            volume = 1.
            search = volume2ball(1,d=0)

        self.pvalue = self.ec_density(search=search, **keywords)

    def resel_estimator(self, n=30, **keywords):
        """
        A quick and dirty resel estimate based on the generator.
        """

        time = Dimension.Dimension('time', length=n)
        warp = Warp.asaffine(self.ugrf.input.warp)
        d = warp.ndim
        indim = [time] + warp.input_coords.dimensions
        incoords = Dimension.Coordinates('voxel', indim)
        outdim = [time] + warp.output_coords.dimensions
        outcoords = Dimension.Coordinates('world', outdim)

        transform = N.zeros((d+2,)*2, N.Float)
        transform[0,0] = 1.0
        transform[1:(d+2),1:(d+2)] = warp.transform

        twarp = Warp.Affine(incoords, outcoords, transform)

        noise = iter(VImage('/tmp/tmpnoise.img', warp=twarp, mode='w'))
        if self.verbose:
            print 'Estimating resels.'

        for i in range(n):
            if self.verbose:
                print 'Generating image [%d] of [%d]' % (i+1, n)
            ugrf = self.ugrf.generate(**keywords)
            noise.next(data=ugrf.image.data)
            del(ugrf); gc.collect()

        noise.warp = twarp
        fwhmest = fastFWHM(noise)
        fwhmest(VImage(noise))
        noise.close()
        os.remove('/tmp/tmpnoise.img'); os.remove('/tmp/tmpnoise.hdr')
        del(noise); gc.collect()

        print fwhmest.integrate(self.search)
        return fwhmest


class ChiSquaredField(GaussianField):

    def __init__(self, image, df, search=None, resels=None, verbose=False, **keywords):
        GaussianField.__init__(self, image, search=search,
                               resels=resels, verbose=verbose, **keywords)
        self.df = df
        self.stat = ChiSquared
        self.getpvalue(m=df)

    def generate(self, **keywords):
        """
        Generate one realization of a chi-squared field.
        """
        value = 0.
        for i in range(self.df):
            tmp = self.ugrf.generate(**keywords).image.data**2
            value += tmp
            del(tmp) ; gc.collect()
        return VImage(value, warp=self.ugrf.input.warp)

class TField(ChiSquaredField):

    def __init__(self, image, df, search=None, resels=None, verbose=False, **keywords):
        GaussianField.__init__(self, image, search=search,
                               resels=resels, verbose=verbose, **keywords)
        self.df = df
        self.stat = Tstat
        self.pvalue = self.getpvalue(m=df)

    def generate(self, **keywords):
        """
        Generate one realization of a T field.
        """
        den = ChiSquaredField.generate(self, **keywords)
        num = GaussianField.generate(self, **keywords)
        value = num.image.data / sqrt(den.image.data / self.df)
        return VImage(value, warp=self.ugrf.input.warp)

class FField(ChiSquaredField):

    def __init__(self, image, df_num, df_den, search=None, resels=None, verbose=False, **keywords):
        GaussianField.__init__(self, image, search=search,
                               resels=resels, verbose=verbose, **keywords)
        self.df_num = df_num
        self.df_den = df_den
        self.stat = Fstat
        self.pvalue = self.getpvalue(m=df_den, n=df_num)

    def generate(self, **keywords):
        """
        Generate one realization of an F field.
        """
        self.df = df_num
        num = ChiSquaredField.generate(self, **keywords)
        self.df = df_den
        den = ChiSquaredField.generate(self, **keywords)

        value = (num.image.data / self.df_num) / (den.image.data / self.df_den)
        return VImage(value, warp=self.ugrf.input.warp)

class ChiBarSquaredField(ChiSquaredField):

    def __init__(self, image, df, search=None, resels=None, verbose=False, **keywords):
        GaussianField.__init__(self, image, search=search,
                               resels=resels, verbose=verbose, **keywords)
        self.stat = ChiBarSquared
        self.df = df
        self.pvalue = self.getpvalue(n=df)

    def generate(self, **keywords):
        """
        Generate one realization of a chi-bar squared field.
        """
        value = 0.

        for i in range(self.df):
            tmp = self.ugrf.generate(**keywords).image.data
            tmp = greater(tmp, 0) * tmp**2
            value += tmp
            del(tmp) ; gc.collect()
        return VImage(value, warp=self.ugrf.input.warp)

