import gc

import numpy as N
from scipy.sandbox.models.utils import recipr

class OneSampleResults(object):
    """
    A container for results from fitting a (weighted) one sample T.
    """

    def __init__(self):
        self.values = {'mean': {'mu': None,
                                'sd': None,
                                't': None,
                                'resid': None,
                                'df_resid': None,
                                'scale': None},
                       'varatio': {'varatio': None,
                                   'varfix': None}}

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, val):
        self.values[key] = val

class OneSample(object):

    def __init__(self, use_scale=True, niter=10, weight_type='sd'):
        if weight_type in ['sd', 'var', 'weight']:
            self.weight_type = weight_type
        else:
            raise ValueError, "Weight type must be one of " \
                  "['sd', 'var', 'weight']"
        self.use_scale = use_scale
        self.niter = niter
        self.value = OneSampleResults()

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
        if W.shape in [(), (1,)]:
            W = N.ones(Y.shape) * W

        nsubject = Y.shape[0]

        if self.value['varatio']['varfix'] is not None:
            sigma2 = N.asarray(self['varatio']['varfix'] *
                               self['varatio']['varatio'])

            if sigma2.shape != ():
                sigma2 = N.multiply.outer(N.ones((nsubject,)), sigma2)
            S = recipr(W) + sigma2
            W = recipr(S)


        mu = N.add.reduce(Y * W, 0) / N.add.reduce(W, 0)
        df_resid = nsubject - 1
        resid = (Y - N.multiply.outer(N.ones(Y.shape[0]), mu)) * N.sqrt(W)

        if self.use_scale:
            scale = N.add.reduce(N.power(resid, 2), 0) / df_resid
        else:
            scale = 1.
        var_total = scale * recipr(N.add.reduce(W, 0))

        self.value['mean']['df_resid'] = df_resid
        self.value['mean']['resid'] = resid
        self.value['mean']['mu'] = mu
        self.value['mean']['sd'] = N.squeeze(N.sqrt(var_total))
        self.value['mean']['t'] = N.squeeze(self.value['mean']['mu'] *
                                            recipr(self.value['mean']['sd']))
        self.value['mean']['scale'] = N.sqrt(scale)

        return self.value

    def estimate_varatio(self, Y, W, df=None):

        Sreduction = 0.99
        S = 1. / W

        nsubject = Y.shape[0]
        df_resid = nsubject - 1

        R = Y - N.multiply.outer(N.ones(Y.shape[0]), N.mean(Y, axis=0))
        sigma2 = N.squeeze(N.add.reduce(N.power(R, 2), axis=0) / df_resid)

        minS = N.minimum.reduce(S, 0) * Sreduction

        Sm = S - N.multiply.outer(N.ones(nsubject), minS)

        for _ in range(self.niter):
            Sms = Sm + N.multiply.outer(N.ones(nsubject), sigma2)
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
            df = N.ones(nsubject)

        df.shape = (1, nsubject)

        _Sshape = S.shape
        S.shape = (S.shape[0], N.product(S.shape[1:]))


        self.value['varatio']['varfix'] = N.dot(df, S) / df.sum()

        S.shape = _Sshape
        self.value['varatio']['varfix'].shape = _Sshape[1:]
        self.value['varatio']['varatio'] = \
                         N.nan_to_num(sigma2 / self.value['varatio']['varfix'])
        return self.value



class OneSampleIterator(object):

    def __init__(self, iterator, outputs=()):
        self.iterator = iter(iterator)
        self.outputs = [iter(output) for output in outputs]


    def weights(self):
        """
        This method should get the weights from self.iterator.
        """
        return 1.

    def _getinputs(self):
        pass

    def fit(self, which='mean', df=None):
        """
        Go through an iterator, instantiating model and passing data,
        going through outputs.
        """

        for data in self.iterator:

            W = self.weights()
            self._getinputs()
            shape = data.shape[1:]

            results = OneSample().fit(data, W, which, df)

            for output in self.outputs:
                out = output.extract(results)
                if output.nout > 1:
                    out.shape = (output.nout,) + shape
                else:
                    out.shape = shape

                output.set_next(data=out)

            del(results);
            gc.collect()

