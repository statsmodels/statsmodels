from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResults, MLEResultsWrapper


class InnnovationModel(MLEModel):
    def __init__(self, endog, k_states, **kwargs):
        kwargs.update(k_posdef=1)
        k_states += 1
        super().__init__(endog, k_states, **kwargs)
        self.ssm['design', 0, -1] = 1
        self.ssm['selection', -1, 0] = 1

    def __setitem__(self, key, value):

        _type = type(key)
        if _type is str:
            if key == 'cov':
                self.ssm['state_cov', 0, 0] = value
            elif key == 'transition':
                self.ssm['transition', :-1, :-1] = value
            elif key == 'selection':
                self.ssm['transition', :-1, -1:] = value
            elif key == 'design':
                self.ssm['design', :, :-1] = value
            elif key in ('state_cov', 'obs_cov'):
                self.ssm['state_cov', :, :] = value
            else:
                return super().__setitem__(key, value)
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if name == 'transition':
                self.ssm['transition', :-1, :-1][slice_] = value
            elif name == 'selection':
                self.ssm['transition', :-1, -1:][slice_] = value
            elif name == 'design':
                self.ssm['design', :, :-1][slice_] = value
            elif name == 'cov':
                self.ssm['state_cov', :, :][slice_] = value
            elif name in ['state_cov', 'obs_cov']:
                self.ssm['state_cov', :, :][slice_] = value
            else:
                return super().__getitem__(key)

    def __getitem__(self, key):

        _type = type(key)
        if _type is str:
            if key == 'transition':
                return self.ssm['transition', :-1, :-1]
            elif key == 'selection':
                return self.ssm['transition', :-1, -1:]
            elif key == 'design':
                return self.ssm['design', :, :-1]
            elif key == 'cov':
                return self.ssm['state_cov', :, :]
            elif key in ('state_cov', 'obs_cov'):
                return self.ssm['state_cov', :, :]
            else:
                return super().__getitem__(key)
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if name == 'transition':
                return self.ssm['transition', :-1, :-1][slice_]
            elif name == 'selection':
                return self.ssm['transition', :-1, -1:][slice_]
            elif name == 'design':
                return self.ssm['design', :, :-1][slice_]
            elif name == 'cov':
                return self.ssm['state_cov', :, :][slice_]
            elif name in ['state_cov', 'obs_cov']:
                return self.ssm['state_cov', :, :][slice_]
            else:
                return super().__getitem__(key)
