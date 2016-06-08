import numpy as np
from collections import OrderedDict

class MarkovSwitchingParams(object):
    def __init__(self, k_regimes):
        self.k_regimes = k_regimes

        self.k_params = 0
        self.k_parameters = OrderedDict()
        self.switching = OrderedDict()
        self.slices_purpose = OrderedDict()
        self.relative_index_regime_purpose = [
            OrderedDict() for i in range(self.k_regimes)]
        self.index_regime_purpose = [
            OrderedDict() for i in range(self.k_regimes)]
        self.index_regime = [[] for i in range(self.k_regimes)]

    def __getitem__(self, key):
        _type = type(key)

        # Get a slice for a block of parameters by purpose
        if _type is str:
            return self.slices_purpose[key]
        # Get a slice for a block of parameters by regime
        elif _type is int:
            return self.index_regime[key]
        elif _type is tuple:
            if not len(key) == 2:
                raise IndexError('Invalid index')
            if type(key[1]) == str and type(key[0]) == int:
                return self.index_regime_purpose[key[0]][key[1]]
            elif type(key[0]) == str and type(key[1]) == int:
                return self.index_regime_purpose[key[1]][key[0]]
            else:
                raise IndexError('Invalid index')
        else:
            raise IndexError('Invalid index')

    def __setitem__(self, key, value):
        _type = type(key)

        if _type is str:
            value = np.array(value, dtype=bool, ndmin=1)
            k_params = self.k_params
            self.k_parameters[key] = (
                value.size + np.sum(value) * (self.k_regimes - 1))
            self.k_params += self.k_parameters[key]
            self.switching[key] = value
            self.slices_purpose[key] = np.s_[k_params:self.k_params]

            for j in range(self.k_regimes):
                self.relative_index_regime_purpose[j][key] = []
                self.index_regime_purpose[j][key] = []

            offset = 0
            for i in range(value.size):
                switching = value[i]
                for j in range(self.k_regimes):
                    # Non-switching parameters
                    if not switching:
                        self.relative_index_regime_purpose[j][key].append(
                            offset)
                    # Switching parameters
                    else:
                        self.relative_index_regime_purpose[j][key].append(
                            offset + j)
                offset += 1 if not switching else self.k_regimes

            for j in range(self.k_regimes):
                offset = 0
                indices = []
                for k, v in self.relative_index_regime_purpose[j].items():
                    v = (np.r_[v] + offset).tolist()
                    self.index_regime_purpose[j][k] = v
                    indices.append(v)
                    offset += self.k_parameters[k]
                self.index_regime[j] = np.concatenate(indices)
        else:
            raise IndexError('Invalid index')
