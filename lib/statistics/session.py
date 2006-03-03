import BrainSTAT
from BrainSTAT.fMRIstat import *
from BrainSTAT.Modules import GLM
from BrainSTAT.RFT.FWHM import *
from numpy.random import *
from pylab import *
import time

IRF = HRF.HRF(deriv=True)

# Specifying frametimes

frametimes = arange(120)*3.
slicetimes = array([0.14, 0.98, 0.26, 1.10, 0.38, 1.22, 0.50, 1.34, 0.62, 1.46, 0.74, 1.58, 0.86])

# Specifying a design matrix using Events

hot = Regressors.PeriodicStimulus(start=9, n=10, duration=9, step=36., name='hot', IRF=IRF)
warm = Regressors.PeriodicStimulus(start=27, n=10, duration=9, step=36., name='warm', IRF=IRF)

design = Design.Design(frametimes=frametimes, regressors = [hot, warm])

# Add temporal trends:

drift = Regressors.SplineConfound(df=7, pad=5., name='drift', window = [0, max(frametimes)])
design.regressors.append(drift)

exclude = 0. * frametimes
exclude[0:3] = 1.
toc = time.time()

# Setup basic model -- includes estimating spatial averages, and AR(1) parameters

infile = BrainSTAT.testfile('test_fmri.hdr')
inimage = BrainSTAT.fMRIImage(infile, repository='/home/jtaylo/.BrainSTAT/repository/')

import gc
import enthought.traits as traits
from BrainSTAT.Modules.KernelSmooth import LinearFilter3d

class Output(traits.HasTraits):

    ndim = traits.Int(1)

    def __init__(self, labels=None, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        if labels is not None:
            self.image = iter(LabelledVImage(zeros(inimage.spatial_warp.shape, Float), labels, warp=inimage.spatial_warp))
        else:
            self.image = iter(VImage(zeros(inimage.spatial_warp.shape, Float), labels, warp=inimage.spatial_warp))

    def __iter__(self):
        return self

    def next(self, data=None, iterator=None):
        self.image.next(data=data)

    def extract(self, results):
        return

class AROutput(Output):

    def extract(self, results):
        resid = results.resid
        rho = add.reduce(resid[0:-1]*resid[1:] / add.reduce(resid[1:-1]**2))
        return rho

class FWHMOutput(Output):

    def __init__(self, fmri, labels=None):
        Output.__init__(self, labels=labels)
        self.ndim = fmri.shape[0]
        self.fwhmest = iter(iterFWHM(fmri.frame(0)))

    def next(self, data=None, iterator=None):
        if hasattr(iterator, 'newslice'): # for LabelledfMRIImage class
            if iterator.newslice:
                self.fwhmest.next(data=iterator.buffer)
        else:
            self.fwhmest.next(data=data)

        del(data) ; gc.collect()

    def extract(self, results):
        return results.norm_resid

class SessionGLM(GLM.GLM):

    def model(self, rho=0., df=None, design=None):

        if hasattr(self.iterator, 'label'):
            rho = self.iterator.label
            model = self.design.model(ARparam=rho, df=df, design=self.dmatrix, covariance=True)
        else:
            model = self.design.model()
        return model

class Session:

    def __init__(self, fmri, design):
        self.fmri = fmri
        self.design = design
        self.dmatrix = design.model()

    def firstpass(self, fwhm=8.0):

        rhoout = AROutput()
        fwhmout = FWHMOutput(self.fmri)

        glm = SessionGLM(self.fmri, self.design, outputs=[rhoout])#, fwhmout])
        glm.fit(resid=True, norm_resid=True)

        kernel = LinearFilter3d(rhoout.image, fwhm=fwhm)
        self.rho = rhoout.image
        self.rho.tofile('rho.img')
        self.fwhm = fwhmout.fwhmest.fwhm

    def secondpass(self):

        self.labels = floor(self.rho * 100.) / 100.
        self.fmri = LabelledfMRIImage(self.fmri, self.labels)

        glm = SessionGLM(self.fmri, self.design)
        glm.fit(resid=True, norm_resid=True)



s = Session(inimage, design)
s.firstpass()

s.rho.view()
s.fwhm.view()


import pylab
pylab.show()

s.secondpass()
