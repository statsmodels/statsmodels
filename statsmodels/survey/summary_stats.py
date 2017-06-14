import numpy as np 
import pandas as pd
#from numbers import Number


class survey_stat(object):

    def __init__(self, data, cluster=None, strata=None, prob_weights=None):

        # preprocessing
        def check_type(data, sample_method):
            """
            converts cluster, strata, etc to an numpy array
            gets rid of column from data if possible

            Parameters
            ----------
            data : array-like
              The raw data
            sample_method : one of cluster, strata, or prob_weight

            Returns
            -------
            A nx1 numpy array
            """

            # check if str, num, or list
            if isinstance(sample_method, str):
                sample = data[sample_method].values
                del data[sample_method]
            elif isinstance(sample_method, int):
                print("int")
                try:
                    sample = data.iloc[:, sample_method].values
                    data.drop(data.columns[sample_method], inplace=True, axis=1)
                except: # data is a ndarray
                    sample = data[:,sample_method]
                    data = np.delete(data, sample_method, 1)
            elif isinstance(sample_method, list):
                sample_method = np.array(sample_method)
            else:
                n = len(sample_method)
                sample_method = sample_method.reshape(n,1)

            return data, sample_method

        data, self.cluster = check_type(data, cluster)
        data, self.strata = check_type(data, strata)
        data, self.prob_weights = check_type(data, prob_weights)
    
        self.data = np.asarray(data)


    # def jackknife(self, col_index):
    #     unique_strata = np.unique(strata)
    #     for i in unique_strata:
    #         idx = np.where(strata == i)
    #         unique_cluster = np.unique()



    def get_subgroup_indices(self):
        combos = {}
        iter = 0
        # add new column that will hold the labels for each subgroup 
        mesh = np.hstack([self.strata, self.cluster, np.zeros([len(self.cluster),1])])
        for i in range(mesh.shape[0]):
            # if we've not seen a particular SSU
            if tuple(mesh[i,[0,1]]) not in combos:
                combos[tuple(mesh[i,[0,1]])] = iter
                mesh[i, 2] = iter
                # the label for that ssu
                iter += 1
            else:
                mesh[i, 2] = combos[tuple(mesh[i,[0,1]])]
        return mesh

    # def show(self):
    #     print(self.data)
    #     print(self.cluster)
    #     print(self.strata)
    #     print(self.prob_weights)

    def survey_total(self):

        n = self.data.shape[0]
        if not isinstance(self.prob_weights, np.ndarray):
            self.prob_weights = np.ones([n,1])
        if not isinstance(self.cluster, np.ndarray):
            self.cluster = np.ones([n,1])
        if not isinstance(self.strata, np.ndarray):
            self.strata = np.ones([n,1])

        # get unique strata, cluster combination. Calculate total for each subgroup
        self.mesh = self.get_subgroup_indices()

        # Only need last column here
        mesh = self.mesh[:,2]
        # for each column, mult obs by prob_weight
        def col_total(self, mesh, index):
            total = [np.dot(self.data[np.where(mesh == i), index], 
                     self.prob_weights[np.where(mesh==i)]) for i in np.unique(mesh)]
            return sum(total)


        self.total = np.array([col_total(self, mesh, j).item() for j in range(self.data.shape[1])])

        return self.total


    def survey_mean(self):
        try:
            self.mean = np.round(self.total / np.sum(self.prob_weights), 2)
        except:
            self.total = self.survey_total()
            self.mean = np.round(self.total / np.sum(self.prob_weights), 2)
        return self.mean
