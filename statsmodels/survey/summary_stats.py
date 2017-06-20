import numpy as np 
import pandas as pd
#from numbers import Number


class SurveyStat(object):

    def __init__(self, data, cluster=None, strata=None, prob_weights=None):

        data, self.cluster = self._check_type(data, cluster)
        data, self.strata = self._check_type(data, strata)
        data, self.prob_weights = self._check_type(data, prob_weights)
        
        self.p = data.shape[1]
        self.data = np.asarray(data)

    # preprocessing to be used by init
    def _check_type(self, data, vname):
        """
        converts cluster, strata, etc to an numpy array
        gets rid of column from data if possible

        Parameters
        ----------
        data : array-like
          The raw data
        vname : one of cluster, strata, or prob_weight

        Returns
        -------
        A nx1 numpy array
        """
        # check if str, num, or list

        self.n = data.shape[0]
        if isinstance(vname, str):
            temp = data[vname].values
            del data[vname]
            vname = temp
            n = len(vname)
            vname = vname.reshape(self.n,1)  
        elif isinstance(vname, list):
            vname = np.array(vname).reshape(self.n,1)
        elif vname is None:
            vname = np.ones([self.n,1])
        else:
            vname = vname.reshape(self.n,1)

        return data, vname


    # def create_subgroup_labels(self):
    #     """
    #     Create unique integer id for each stratum x cluster combination
        
    #     The ids are 0, 1, ... and are kept as an attribute
    #     """

    #     _, i1 = np.unique(self.strata, return_inverse=True)
    #     _, i2 = np.unique(self.cluster, return_inverse=True)
    #     labels = i1 * len(i2) + i2
    #     self.grp_ix = dict.fromkeys(np.unique(labels),[])
    #     # doesnt append correctly
    #     for i, k in enumerate(labels):
    #         self.grp_ix[k].append(i)
    #     return self.group_ix

    # def get_psu_indices(self):
    #     self.indices_dict = {}
    #     unique_strata = np.unique(self.strata)
    #     stratum_stats = np.empty(len(unique_strata))
    #     for id, stratum in enumerate(unique_strata):
    #         # get indices for a particular stratum and the # of clusters in it
    #         id_stratum = np.where(self.strata == stratum)[0]        
    #         unique_clusters = np.unique(self.cluster[id_stratum])
    #         num_clusters = len(unique_clusters)
    #         self.indices_dict[stratum] = unique_clusters
    #     # print(self.indices_dict)

    def _jackknife(self, method, column_index, object):
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
        # self.get_psu_indices()
        # stratum_stats = np.empty(len(self.indices_dict.keys()))
        # for id, stratum in enumerate(self.indices_dict.keys()):
        #     id_stratum = np.where(self.strata == stratum)[0]
        #     num_clusters = len(self.indices_dict[stratum])
        #     cluster_stats = np.empty(num_clusters)
        #     for ind, cluster in enumerate(self.indices_dict[key]):
                new_weights = self._reweight(cluster, id_stratum, method)

                cluster_stats[ind] = object._stat(new_weights)

            stratum_stats[id] = np.sum((cluster_stats) ** 2)
            stratum_stats[id] *= (num_clusters - 1) / num_clusters
        return np.sqrt(np.sum(stratum_stats))

            # diff = (cluster_stats - method) * (num_clusters) / (num_clusters - 1)
            # stratum_stats[id] = sum(diff)
        

    def _reweight(self, cluster, id_stratum, method):
        # make sure to throw and error if len(num_clusters == 1) 
        # ie you can't calc a statistic minus a cluster bc there's only one
        
        num_clusters = len(np.unique(self.cluster[id_stratum]))

        id_noncluster = np.intersect1d(id_stratum, np.where(self.cluster != cluster)[0])

        # in stratum h and in cluster j
        id_cluster = np.intersect1d(id_stratum, np.where(self.cluster == cluster)[0])

        new_weights = np.array(self.prob_weights, copy=True)
        new_weights[id_noncluster] = (num_clusters / (num_clusters - 1)) * self.prob_weights[id_noncluster]
        new_weights[id_cluster] = 0

        return new_weights        

    def _stat(self):
        raise NotImplementedError

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


        self._total = [np.dot(self.data[:, index], self.prob_weights).item() for index in range(self.data.shape[1])]
        self._total = np.array(self._total)

        if method == "jack":
            total_se = np.array([self._jackknife("total", index, SurveyTotal(self.data, self.cluster, self.strata, self.prob_weights,index,self._total)) for index in range(self.data.shape[1])])
        
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
            mean_se = np.array([self._jackknife("mean", index, SurveyMean(self.data, self.cluster, self.strata, self.prob_weights,index, self._mean)) for index in range(self.data.shape[1])])
        
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


class SurveyMean(SurveyStat):

    def __init__(self, data, cluster, strata, prob_weights, column_index,parent_stat):
        SurveyStat.__init__(self, data, cluster, strata, prob_weights)
        self.column_index = column_index
        self.parent_stat = parent_stat

    def _stat(self, new_weights):
        stat = np.array(np.dot(self.data[:, self.column_index], new_weights).item())
        stat /= np.sum(new_weights)
        stat -= self.parent_stat[self.column_index]
        return stat

class SurveyTotal(SurveyStat):

    def __init__(self, data, cluster, strata, prob_weights, column_index, parent_stat):
        SurveyStat.__init__(self, data, cluster, strata, prob_weights)
        self.column_index = column_index
        self.parent_stat = parent_stat

    def _stat(self, new_weights):
        stat = np.array(np.dot(self.data[:, self.column_index], new_weights).item())
        stat -= self.parent_stat[self.column_index]
        return stat




df = pd.read_csv("~/Documents/survey_data")
df.drop("Unnamed: 0", inplace=True, axis=1)
# df = df.loc[df['dnum'].isin([637,437])]
print(df.head())

test = SurveyStat(data=df, cluster="dnum", prob_weights="pw")
# # test.show()
# print("\n \n \n")
survey_tot = test.total('jack') # matches perfectly with R result
print(survey_tot)
survey_mean = test.mean('jack') # SE is bigger than R's result
# i did my method in R and got what I got here... thus, R must be doing something differently for their mean
print(survey_mean)
# print(test.percentile(25)) # R seems to take the difference between the the (i-1)th and ith value when I dont.
# print(test.median()) # matches R
