import numpy as np 
import pandas as pd
#from numbers import Number


class SurveyStat(object):

    def __init__(self, data, cluster=None, strata=None, prob_weights=None):

        data, self.cluster = self._check_type(data, cluster)
        data, self.strata = self._check_type(data, strata)
        data, self.prob_weights = self._check_type(data, prob_weights)
        
        n = data.shape[0]
        p = data.shape[1]
        self.data = np.asarray(data).reshape(n,p)

    # preprocessing to be used by init
    def _check_type(self, data, sampling_method):
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


    def _create_subgroup_labels(self):
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

        _, i1 = np.unique(strata, return_inverse=True)
        _, i2 = np.unique(clusters, return_inverse=True)
        labels = i1 * len(i2) + i2
        return labels

    def _jackknife(self, method, column_index):
        unique_strata = np.unique(self.strata)
        stratum_stats = np.empty(len(unique_strata))
        for id, stratum in enumerate(unique_strata):
            # get indices for a particular stratum and the # of clusters in it
            id_stratum = np.where(self.strata == stratum)[0]        
            unique_clusters = np.unique(self.cluster[id_stratum])
            num_clusters = len(unique_clusters)

            # get the statistic without a particular cluster
            cluster_stats = np.empty(num_clusters)

            for ind, cluster in enumerate(unique_clusters):
                new_weights = self._reweight(cluster, id_stratum, method)
                if method == "total":
                    cluster_stats[ind] = np.array(np.dot(self.data[:, column_index], new_weights).item())
                    cluster_stats[ind] -= self._total[column_index]
                elif method == "mean":
                    cluster_stats[ind] = np.array(np.dot(self.data[:, column_index], new_weights).item())
                    cluster_stats[ind] /= np.sum(new_weights)
                    cluster_stats[ind] -= self._mean[column_index]

            stratum_stats[id] = np.sum((cluster_stats) ** 2)
            stratum_stats[id] *= (num_clusters - 1) / num_clusters
        return np.sqrt(np.sum(stratum_stats))

            # diff = (cluster_stats - method) * (num_clusters) / (num_clusters - 1)
            # stratum_stats[id] = sum(diff)
        

    def _reweight(self, cluster, id_stratum, method):
        # make sure to throw and error if len(num_clusters == 1) 
        # ie you can't calc a statistic minus a cluster bc there's only one
        num_clusters = len(np.unique(self.cluster[id_stratum]))
        # in stratum h but not in cluster j
        # print("cluster #:", cluster)
        # print("cluster indices:", np.where(self.cluster == cluster)[0])
        # print("stratum indices:", id_stratum)
        id_noncluster = np.intersect1d(id_stratum, np.where(self.cluster != cluster)[0])

        # in stratum h and in cluster j
        id_cluster = np.intersect1d(id_stratum, np.where(self.cluster == cluster)[0])

        new_weights = np.array(self.prob_weights, copy=True)
        new_weights[id_noncluster] = (num_clusters / (num_clusters - 1)) * self.prob_weights[id_noncluster]
        new_weights[id_cluster] = 0

        return new_weights        


    def show(self):
        print(self.data)
        print(self.cluster)
        print(self.strata)
        print(self.prob_weights)

    def total(self, method):
        """
        Calculates the total for each variable. 

        Parameters
        ----------
        self

        Returns
        -------
        A tuple of length 2, each index is a 1xp array. The first array holds 
        the total for each of the p vars in self.data. The second array is the
        SE for each total
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

        self._total = [np.dot(self.data[:, index], self.prob_weights).item() for index in range(self.data.shape[1])]
        self._total = np.array(self._total)

        if method == "jack":
          total_se = np.array([self._jackknife("total", index) for index in range(self.data.shape[1])])
        
        return self._total, total_se


    def mean(self, method):
        """
        Calculates survey mean

        Parameters
        ----------
        self

        Returns
        -------
        A tuple of length 2, each index is a 1xp array. The first array holds 
        the mean for each of the p vars in self.data. The second array is the
        SE for each total 
        """
        try:
            self._mean = np.round(self._total / np.sum(self.prob_weights), 2)
        # if self.total doesn't exist yet
        except AttributeError:
            self._total = self.total()
            self._mean = self._total / np.sum(self.prob_weights)

        if method == "jack":
            mean_se = np.array([self._jackknife("mean", index) for index in range(self.data.shape[1])])
        
        return self._mean, mean_se

    # need to make this work when input is an array-like object
    def percentile(self, percentile): 
        p = self.data.shape[1]
        self._percentile = np.empty(p)
        
        for index in range(p):
            sorted_weights = [x for (y,x) in sorted(zip(self.data[:, index],self.prob_weights))]
            sorted_weights = np.array(sorted_weights)
            cumsum_weights = np.cumsum(sorted_weights)

            perc = (cumsum_weights[-1] * percentile) / 100
            sorted_data = np.sort(self.data[:, index])
            
            if perc in cumsum_weights:
                ind = np.where(cumsum_weights == perc)[0].item()
                self._percentile[index] = (sorted_data[ind] + sorted_data[ind + 1]) / 2
            else:
                ind = np.argmax(cumsum_weights > perc)
                self._percentile[index] = sorted_data[ind]
        return self._percentile

    def median(self):
        self._median = self.percentile(50)
        return self._median




# df = pd.read_csv("~/Documents/survey_data")
# df.drop("Unnamed: 0", inplace=True, axis=1)
# df = df.loc[df['dnum'].isin([637,437])]
# print(df.head())

# test = SurveyStat(data=df, cluster="dnum", prob_weights="pw")
# # test.show()
# print("\n \n \n")
# survey_tot = test.total('jack') # matches perfectly with R result
# print(survey_tot)
# survey_mean = test.mean('jack') # SE is bigger than R's result
# # i did my method in R and got what I got here... thus, R must be doing something differently for their mean
# print(survey_mean)
# print(test.percentile(25)) # R seems to take the difference between the the (i-1)th and ith value when I dont.
# print(test.median()) # matches R