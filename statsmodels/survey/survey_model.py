import numpy as np

class SurveyModel(object):

    def __init__(self, design, model_class, init_args={}, fit_args={}):
        self.design = design
        self.model = model_class
        self.init_args = dict(init_args)
        self.fit_args = dict(fit_args)

    def fit(self, y, X, replicates=None):
        self.init_args["weights"] = self.design.weights
        self.params = self._get_params(y, X)
        if replicates is None:
            try:
                k = self.design.nclust
            except AttributeError:
                k = self.design.rep_weights.shape[1]
        else:
            k = replicates
        self.replicate_params = []
        for c in range(k):
            w = self.design.get_rep_weights(c=c)
            self.init_args["weights"] = w
            print('weights', self.init_args['weights'])
            model = self.model(y, X, **self.init_args)
            self.replicate_params.append(self._get_params(y, X))
        self.replicate_params = np.asarray(self.replicate_params)
        print('new params', self.replicate_params)

        self.replicate_params -= self.params

        # for now, just working with jackknife to see if it works
        if self.design.cov_method == 'jack':
            try:
                nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
                mh = np.sqrt((nh - 1) / nh)
                self.replicate_params = mh[:, None] * self.replicate_params
            except AttributeError:
                nh = self.design.rep_weights.shape[1]
                mh = np.sqrt((nh - 1) / nh)
                self.replicate_params *= mh
            self.vcov = np.dot(self.replicate_params.T, self.replicate_params)
            self.stderr = np.sqrt(np.diag(self.vcov))

    def _get_params(self, y, X):
        model = self.model(y, X, **self.init_args)
        result = model.fit(**self.fit_args)
        return result.params

