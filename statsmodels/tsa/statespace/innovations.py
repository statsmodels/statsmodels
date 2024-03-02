from statsmodels.tsa.statespace.mlemodel import MLEModel


class InnnovationsMLEModel(MLEModel):
    def __init__(self, endog, k_states, **kwargs):
        kwargs.update(k_posdef=1)
        # hidden states
        k_states += 1
        super().__init__(endog, k_states, **kwargs)
        self.ssm['design', 0, 0] = 1
        self.ssm['selection', 0, 0] = 0

    def __setitem__(self, key, value):

        _type = type(key)
        if _type is str:
            if key == 'cov':
                self.ssm['state_cov', 0, 0] = value
            elif key == 'F':
                self.ssm['transition', 1:, 1:] = value
            elif key == 'g':
                self.ssm['transition', 0, 1:] = value
            elif key == 'w':
                self.ssm['design', 0, 1:] = value
            else:
                return super().__setitem__(key, value)
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if key == 'name':
                self.ssm['state_cov', 0, 0] = value
            elif key == 'F':
                self.ssm['transition', 1:, 1:][slice_] = value
            elif key == 'g':
                self.ssm['transition', 0, 1:][slice_] = value
            elif key == 'w':
                self.ssm['design', 0, 1:][slice_] = value
            else:
                return super().__setitem__(name, slice_)

    def __getitem__(self, key):

        _type = type(key)
        if _type is str:
            if key == 'g':
                return self.ssm['transition', 1:, 0:1]
            elif key == 'F':
                return self.ssm['transition', 1:, 1:]
            elif key == 'w':
                return self.ssm['design', :, 1:]
            elif key == 'cov':
                return self.ssm['state_cov', :, :]
            else:
                return super().__getitem__(key)
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if name == 'g':
                return self.ssm['transition', 1:, 0:1][slice_]
            elif name == 'F':
                return self.ssm['transition', 1:, 1:][slice_]
            elif name == 'w':
                return self.ssm['design', :, 1:][slice_]
            elif name == 'cov':
                return self.ssm['state_cov', :, :][slice_]
            else:
                return super().__getitem__(key)