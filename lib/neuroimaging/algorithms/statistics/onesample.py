import gc

import numpy as N
from scipy.sandbox.models.utils import recipr
from neuroimaging import traits

class OneSampleResults(object):
    """
    A container for results from fitting a (weighted) one sample T.

    """

    def __init__(self):
        raise NotImplementedError
        self.values = {'mean': {'mu': None,
                                'sd': None,
                                't': None,
                                'resid': None,
                                'df_resid': None,
                                'scale': None},
                       'varatio': {'varatio': None,
                                   'varfix': None}}

class OneSample(traits.HasTraits):

    weight_type = traits.Trait('sd', 'var', 'weight')
    varatio = traits.Trait(traits.Any())
    varfix = traits.Trait(traits.Any())
    niter = traits.Int(10)
    use_scale = traits.true

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)

    def estimate_varatio(self, Y, W, df=None):

        Sreduction = 0.99
        S = 1. / W

        nsubject = Y.shape[0]
        df_resid = nsubject - 1

        R = Y - N.multiply.outer(N.ones(Y.shape[0]), N.mean(Y, axis=0))
        sigma2 = N.squeeze(N.add.reduce(N.power(R, 2), axis=0) / df_resid)

        minS = N.minimum.reduce(S, 0) * Sreduction

        Sm = S - N.multiply.outer(N.ones((nsubject,), N.float64), minS)

        for _ in range(self.niter):
            Sms = Sm + N.multiply.outer(N.ones((nsubject,), N.float64), sigma2)
            W = recipr(Sms)
            Winv = 1. / N.add.reduce(W, axis=0)
            mu = Winv * N.add.reduce(W * Y, axis=0)
            R = W * (Y - N.multiply.outer(N.ones(nsubject), mu))
            ptrS = 1 + N.add.reduce(Sm * W, 0) - \
                   N.add.reduce(Sm * N.power(W, 2), axis=0) * Winv
            sigma2 = N.squeeze((sigma2 * ptrS + N.power(sigma2, 2) *
                                N.add.reduce(N.power(R,2), 0)) / nsubject)

        sigma2 = sigma2 - minS

        if df is None:
            df = N.ones((nsubject,), N.float64)

        df.shape = (1, nsubject)

        _Sshape = S.shape
        S.shape = (S.shape[0], N.product(S.shape[1:]))

        value = OneSampleResults()
        value['varatio']['varfix'] = N.dot(df, S) / df.sum()

        S.shape = _Sshape
        value['varatio']['varfix'].shape = _Sshape[1:]
        value['varatio']['varatio'] = N.nan_to_num(sigma2 / value.varfix)
        return value

    def fit(self, Y, W, which='mean', df=None):
        if which == 'mean':
            return self.estimate_mean(Y, W)
        else:
            return self.estimate_varatio(Y, W, df=df)

    def get_weights(self, W):
        try:
            if W.ndim == 1:
                W.shape = (W.shape[0], 1)
        except:
            pass

        if self.weight_type == 'sd':
            W = 1. / N.power(W, 2)
        elif self.weight_type == 'var':
            W = 1. / W
        return N.asarray(W)

    def estimate_mean(self, Y, W):

        if Y.ndim == 1:
            Y.shape = (Y.shape[0], 1)
        W = self.get_weights(W)
        if W.shape in [(),(1,)]:
            W = N.ones(Y.shape) * W

        nsubject = Y.shape[0]

        if self.varfix is not None:
            sigma2 = N.asarray(self.varfix * self.varatio)

            if sigma2.shape != ():
                sigma2 = N.multiply.outer(N.ones((nsubject,), N.float64), sigma2)
            S = recipr(W) + sigma2
            W = recipr(S)


        mu = N.add.reduce(Y * W, 0) / N.add.reduce(W, 0)

        value = OneSampleResults()
        value['mean']['df_resid'] = Y.shape[0] - 1
        value['mean'['resid'] = (Y - N.multiply.outer(N.ones(Y.shape[0], N.float64), mu)) * N.sqrt(W)

        if self.use_scale:
            scale = N.add.reduce(N.power(value.resid, 2), 0) / value.df_resid
        else:
            scale = 1.
        var_total = scale * recipr(N.add.reduce(W, 0))

        value['mean']['mu'] = mu
        value['mean']['sd'] = N.squeeze(N.sqrt(var_total))
        value['mean']['t'] = N.squeeze(value.mu * recipr(value.sd))
        value['mean']['scale'] = N.sqrt(scale)

        return value

class OneSampleIterator(object):

    def __init__(self, iterator, outputs=[]):
        self.iterator = iter(iterator)
        self.outputs = [iter(output) for output in outputs]

    def weights(self):
        """
        This method should get the weights from self.iterator.
        """
        return 1.

    def fit(self, **keywords):
        """
        Go through an iterator, instantiating model and passing data,
        going through outputs.
        """

        for data in self.iterator:

            W = self.weights()
            shape = data.shape[1:]

            results = OneSample.fit(self, data, W, **keywords)

            for output in self.outputs:
                out = output.extract(results)
                if output.nout > 1:
                    out.shape = (output.nout,) + shape
                else:
                    out.shape = shape

                output.next(data=out)

            del(results);
            gc.collect()

