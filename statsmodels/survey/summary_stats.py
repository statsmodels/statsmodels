import numpy as np 
import pandas as pd
#from numbers import Number


class SurveyStat(object):

    def __init__(self, data, cluster=None, strata=None, prob_weights=None):

        data, self.cluster = self.__check_type(data, cluster)
        data, self.strata = self.__check_type(data, strata)
        data, self.prob_weights = self.__check_type(data, prob_weights)
    
        self.data = np.asarray(data)

    # preprocessing to be used by init
    def __check_type(self, data, sampling_method):
        """
        converts cluster, strata, etc to an numpy array
        gets rid of column from data if possible

        Parameters
        ----------
        data : array-like
          The raw data
        sampling_method : one of cluster, strata, or prob_weight

        Returns
        -------
        A nx1 numpy array
        """

        # check if str, num, or list
        if isinstance(sampling_method, str):
            temp = data[sampling_method].values
            del data[sampling_method]
            sampling_method = temp
            n = len(sampling_method)
            sampling_method = sampling_method.reshape(n,1)
        elif isinstance(sampling_method, int):
            try:
                temp = data.iloc[:, sampling_method].values
                data.drop(data.columns[sampling_method], inplace=True, axis=1)
                sampling_method = temp
                n = len(sampling_method)
                sampling_method = sampling_method.reshape(n,1)
            except AttributeError: # data is a ndarray
                temp = data[:,sampling_method]
                data = np.delete(data, sampling_method, 1)
                sampling_method = temp
                n = len(sampling_method)
                sampling_method = sampling_method.reshape(n,1)
        elif isinstance(sampling_method, list):
            sampling_method = np.array(sampling_method)
        elif sampling_method is None:
            n = data.shape[0]
            self.sampling_method = np.ones([n,1])
        else:
            n = len(sampling_method)
            sampling_method = sampling_method.reshape(n,1)

        return data, sampling_method

    # def jackknife(self, col_index):
    #     unique_strata = np.unique(strata)
    #     for i in unique_strata:
    #         idx = np.where(strata == i)
    #         unique_cluster = np.unique()



    def __create_subgroup_labels(self):
        """
        creates labels for each SSU within a PSU so that one is sure that 
        operations are being done withing each sampling subgroup

        Parameters
        ----------
        self

        Returns
        -------
        A nx3 array, the first column being self.strata, the second column
        is self.cluster, the third column is the unique label for each subgroup
        """

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

    def add_prob_weights(self, prob_weights):
        """
        Allows user to pass in a the probability weights either as a an array-like object
        or as an integer to represent a column in their data. 
        *issue: what if their original data was a DataFrame and they input an column from the 
        df? This is an issue bc check_type won't delete the column from self.data.strata

        Parameters
        ----------
        prob_weights: An array-like obj with len() = n or or column index

        Returns
        -------
        A nx3 array, the first column being self.strata, the second column
        is self.cluster, the third column is the unique label for each subgroup
        """

        self.data, prob_weights = self.__check_type(data, strata)


    def add_strata(self, strata):
        self.data, self.strata = self.__check_type(data, strata)

    def add_cluster(self, cluster):
        self.data, self.cluster = self.__check_type(data, cluster)


    def show(self):
        print(self.data)
        print(self.cluster)
        print(self.strata)
        print(self.prob_weights)

    def survey_total(self):
        """
        Calculates the total for each variable. 

        Parameters
        ----------
        self

        Returns
        -------
        A 1xp array, holding the values for each of the p vars in self.data
        """

        # creates "dummy" arrays for prob_weights, cluster, and strata
        # if any or all were not supplied
        n = self.data.shape[0]
        if not isinstance(self.prob_weights, np.ndarray):
            self.prob_weights = np.ones([n,1])
        if not isinstance(self.cluster, np.ndarray):
            self.cluster = np.ones([n,1])
        if not isinstance(self.strata, np.ndarray):
            self.strata = np.ones([n,1])

        # get unique strata, cluster label. Calculate total for each subgroup
        # self.mesh = self.__create_subgroup_labels()

        # # Only need last column here
        # mesh = self.mesh[:,2]

        # # for each column and in each subgroup, mult observation by prob_weight
        # def col_total(self, mesh, index):
        #     total = [np.dot(self.data[np.where(mesh == i), index], 
        #              self.prob_weights[np.where(mesh==i)]) for i in np.unique(
        #              mesh)]
        #     return sum(total)


        # self.total = np.array([col_total(self, mesh, j).item() for j in range(
        #     self.data.shape[1])])
        self.total = [np.dot(self.data[:, index], self.prob_weights).item(
            ) for index in range(self.data.shape[1])]
        self.total = np.array(self.total)
        return self.total


    def survey_mean(self):
        """
        Calculates survey mean

        Parameters
        ----------
        self

        Returns
        -------
        A 1xp array, containg the mean for each of the p vars in self.data
        """

        try:
            self.mean = np.round(self.total / np.sum(self.prob_weights), 2)
        # if self.total doesn't exist yet
        except AttributeError:
            self.total = self.survey_total()
            self.mean = np.round(self.total / np.sum(self.prob_weights), 2)
        return self.mean


    def survey_percentile(self, percentile):
        cumsum_weights = np.cumsum(self.prob_weights)
        perc = (cumsum_weights[-1] * percentile) / 100
        p = self.data.shape[1]
        if perc in cumsum_weights:
            index = np.where(cumsum_weights == perc)[0].item()
            self.percentile = np.array([(self.data[index, var] + self.data[index + 1,var]) / 2 for var in range(p)])
        else:
            index = np.argmax(cumsum_weights > perc)
            self.percentile = np.array([self.data[index, var] for var in range(p)])
        return self.percentile

    def survey_median(self):
        self.median = self.survey_percentiles(50)
        return self.median