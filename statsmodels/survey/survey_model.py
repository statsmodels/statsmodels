import numpy as np

class SurveyModel(object):

    def __init__(self, design, model_class, init_args={}, fit_args={}):
        self.design = design
        self.model = model_class
        self.init_args = dict(init_args)
        self.fit_args = dict(fit_args)

    def _centering(self, array=None):
        if self.center_by == 'est':
            array -= self.params
        elif self.center_by == 'global':
            array -= array.mean(0)
        elif self.center_by == 'stratum':
            if self.design.rep_weights is None:
                for s in range(self.design.nstrat):
                    # center the 'delete 1' statistic
                    array[self.design.ii[s], :] -= array[self.design.ii[s],
                                                         :].mean(0)
        else:
            raise ValueError("Centering option not implemented")
        return array

    def fit(self, y, X, cov_method='jack', center_by='est', replicates=None):
        self.center_by = center_by
        self.init_args["weights"] = self.design.weights
        self.params = self._get_params(y, X)
        if replicates is None:
            k = self.design.nclust
        else:
            k = replicates

        replicate_params = []
        for c in range(k):
            w = self.design.get_rep_weights(cov_method=cov_method, c=c)
            self.init_args["weights"] = w
            print('weights', self.init_args['weights'])
            replicate_params.append(self._get_params(y, X))

        replicate_params = np.asarray(replicate_params)
        print('new params', replicate_params)
        self.replicate_params = self._centering(replicate_params)

        # for now, just working with jackknife to see if it works
        if cov_method == 'jack':
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

# import pandas as pd
# df = pd.read_stata("/home/jarvis/Downloads/nhanes2.dta")
# y = df["weight"]
# X = df["height"]
# design = SurveyDesign(strata=df["strata"], cluster=df["psu"], weights=df['weight'])
# mod = SurveyModel(design, sm.WLS)
# mod.fit(y, X)