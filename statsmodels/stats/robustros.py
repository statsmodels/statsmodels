from __future__ import print_function, division

import numpy as np
from scipy import stats

from statsmodels.graphics import utils
import pandas as pd


def _ros_sort(dataframe, result='res', censorship='cen'):
    '''
    This function prepares a dataframe for ROS. It sorts ascending with
    left-censored observations on top.

    Parameters
    ----------
    dataframe : a pandas dataframe with results and qualifiers.
        The qualifiers of the dataframe must have two states:
        detect and non-detect.
    result (default = 'res') : name of the column in the dataframe
        that contains result values.
    censorship (default = 'cen') : name of the column in the dataframe
        that indicates that a result is left-censored.
        (i.e., True -> censored, False -> uncensored)

    Output
    ------
    Sorted pandas DataFrame.

    '''

    # separate uncensored data from censored data
    censored = dataframe[dataframe[censorship]].sort(columns=result)
    uncensored = dataframe[~dataframe[censorship]].sort(columns=result)

    return censored.append(uncensored)

class RobustROSEstimator(object):
    '''
    Class to implement the Robust regression-on-order statistics (ROS)
    method outlined in Nondetects and Data Analysis by Dennis R. Helsel
    (2005) to estimate the left-censored (non-detect) values of a
    dataset.

    Parameters
    ----------
    data : pandas DataFrame or None
        The censored dataset for which the non-detect values need to be
        estimated. If None, `result` and `censorship` must be sequences.
    result : optional string (default='res') or sequence.
        The name of the column containing the numerical values of the
        dataset. Left-censored values should be set to the detection
        limit. If `data` is None, this should be a sequence of reported
        values.
    censorship : optional string (default='cen') or sequence
        The name of the column containing indicating which observations
        are censored. If `data` is None, this should be a sequence of
        values that, when evaluated as booleans, imply the following:
        `True` implies Left-censorship. `False` -> uncensored.
    transform_in : optional function or None (default = np.log)
        By default, lognormality is assumed and the linear regression
        is performed on log-transformed data. An arbitrary transformation
        function can be passed using this parameter. If `None` is passed,
        the data will remain untransformed.
    transform_out : optional function or None (default = np.exp)
        This is used after the linear regression to revert the transformation
        performed by `transform_in`. If `None` is passed, the data will remain
        untransformed.

    Attributes
    ----------
    nobs : int
        Total number of results in the dataset
    ncen : int
        Total number of non-detect results in the dataset.
    cohn : pandas DataFrame
        A DataFrame of the unique detection limits found in `data` along
        with the `A`, `B`, `C`, and `PE` quantities computed by the
        estimation.
    data : pandas DataFrame
        An expanded version of the original dataset `data` passed the
        constructor included in the `modeled` column.
    debug : pandas DataFrame
        A full version of the `data` DataFrame that includes preliminary
        quantities.

    Example
    -------
    >>> from statsmodels.stats.robustros import RobustROSEstimator
    >>> ros = RobustROSEstimator(myDataFrame, result='conc',
                                 censorship='censored')
    >>> ros.estimate()
    >>> print(ros.data)

    Notes
    -----
    It is inappropriate to replace specific left-censored values with
    the estimated values from this method. The estimated values
    (self.data['modeled']) should instead be used to refine descriptive
    statistics of the dataset as a whole.

    Also, this code is still considered expirmental and it's APIs may
    change dramatically in subsequent releases.

    '''

    def __init__(self, data=None, result='res', censorship='cen',
                 min_uncensored=2, max_fraction_censored=0.8,
                 transform_in=np.log, transform_out=np.exp):

        self.min_uncensored = min_uncensored
        self.max_fraction_censored = max_fraction_censored

        # confirm a datatype real quick
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame({'res': result, 'cen': censorship})
            except:
                msg = ("Input `data` must be a pandas DataFrame or "
                       "`result` and `censorship` must be sequences")
                raise ValueError(msg)

            self.result_name = 'res'
            self.censorship_name = 'cen'
        else:
            self.result_name = result
            self.censorship_name = censorship

        if not data.index.is_unique:
            raise ValueError("Index of input DataFrame `data` must be unique")

        try:
            np.float64(data[self.result_name])
        except ValueError:
            raise ValueError('Result data is not uniformly numeric')

        if data[self.result_name].min() <= 0:
            raise ValueError('All result values of `data` must be positive')

        # pre- and post-regression transformations
        self.transform_in = (
            transform_in if transform_in is not None else lambda x: x
        )
        self.transform_out = (
            transform_out if transform_out is not None else lambda x: x
        )

        # and get the basic info
        self.nobs = data.shape[0]
        self.ncen = data[data[self.censorship_name]].shape[0]

        self._raw_data = data

        # sort the data, selecting only the results and censorship columns
        self.data = data[[self.result_name, self.censorship_name]]
        self.data = _ros_sort(self.data, result=self.result_name,
                              censorship=self.censorship_name)

        # create a dataframe of detection limits and their parameters
        # used in the ROS estimation
        self.cohn = self._get_cohn_numbers()

    def _get_cohn_numbers(self):
        '''
        Computes the Cohn numbers for the detection limits in the dataset.

        The Cohn Numbers are:
            + A_j = the number of uncensored obs above the jth threshold.
            + B_j = the number of observations (cen & uncen) below the jth
              threshold.
            + C_j = the number of censored observations at the jth threshold
            + PE_j = the probability of exceeding the jth threshold
            + detection_limit = unique detection limits in the dataset.
            + lower -> a copy of the detection_limit column
            + upper -> lower shifted down 1 step
        '''

        def nuncen_above(row):
            '''
            The number of uncensored obs above the given threshold. (A_j)
            '''
            # index of results above the lower DL
            above = self.data[self.result_name] >= row['lower']

            # index of results below the upper DL
            below = self.data[self.result_name] < row['upper']

            # index of non-detect results
            detect = self.data[self.censorship_name] == False

            # return the number of results where all conditions are True
            return self.data[above & below & detect].shape[0]

        def nobs_below(row):
            '''
            The number of observations (cen & uncen) below the given
            threshold. (B_j)
            '''
            # index of data less than the lower DL
            less_than = self.data[self.result_name] < row['lower']

            # index of data less than or equal to the lower DL
            less_thanequal = self.data[self.result_name] <= row['lower']

            # index of detects, non-detects
            uncensored = self.data[self.censorship_name] == False
            censored = self.data[self.censorship_name] == True

            # number results less than or equal to lower DL and non-detect
            LTE_censored = self.data[less_thanequal & censored].shape[0]

            # number of results less than lower DL and detected
            LT_uncensored = self.data[less_than & uncensored].shape[0]

            # return the sum
            return LTE_censored + LT_uncensored

        def ncen_equal(row):
            '''
            The number of censored observations at the given threshold (C_j)
            '''
            censored_index = self.data[self.censorship_name]
            censored_data = self.data[self.result_name][censored_index]
            censored_below = censored_data == row['lower']
            return censored_below.sum()

        # unique values
        censored_data = self.data[self.censorship_name]
        cohn = pd.unique(self.data[self.result_name][censored_data])

        # if there is a results smaller than the minimum detection limit,
        # add that value to the array
        if cohn.shape[0] > 0:
            if self.data[self.result_name].min() < cohn.min():
                cohn = np.hstack([self.data[self.result_name].min(), cohn])

            # create a dataframe
            cohn = pd.DataFrame(cohn, columns=['DL'])

            # copy the cohn in two columns. offset the 2nd (upper) column
            cohn['lower'] = cohn['DL']
            if cohn.shape[0] > 1:
                cohn['upper'] = cohn.DL.shift(-1).fillna(value=np.inf)
            else:
                cohn['upper'] = np.inf

            # compute A, B, and C
            cohn['nuncen_above'] = cohn.apply(nuncen_above, axis=1)
            cohn['nobs_below'] = cohn.apply(nobs_below, axis=1)
            cohn['ncen_equal'] = cohn.apply(ncen_equal, axis=1)

            # add an extra row
            cohn = cohn.reindex(range(cohn.shape[0]+1))

            # add the 'prob_exceedance' column, initialize with zeros
            cohn['prob_exceedance'] = 0.0

        else:
            dl_cols = ['DL', 'lower', 'upper', 'nuncen_above',
                       'nobs_below', 'ncen_equal', 'prob_exceedance']
            cohn = pd.DataFrame(np.empty((0,7)), columns=dl_cols)

        return cohn

    def _compute_plotting_positions(self):
        def _ros_plotting_pos(row):
            '''
            Helper function to compute the ROS'd plotting position
            '''
            dl_1 = self.cohn.iloc[row['det_limit_index']]
            dl_2 = self.cohn.iloc[row['det_limit_index']+1]
            if row[self.censorship_name]:
                return (1 - dl_1['prob_exceedance']) * row['rank']/(dl_1['ncen_equal']+1)
            else:
                return (1 - dl_1['prob_exceedance']) + (dl_1['prob_exceedance'] - dl_2['prob_exceedance']) * \
                        row['rank'] / (dl_1['nuncen_above']+1)

        self.data['plot_pos'] = self.data.apply(_ros_plotting_pos, axis=1)

        # correctly sort the plotting positions of the ND data:
        ND_plotpos = self.data['plot_pos'][self.data[self.censorship_name]]
        ND_plotpos.values.sort()
        self.data.loc[self.data[self.censorship_name], 'plot_pos'] = ND_plotpos

    def estimate(self):
        '''
        Estimates the values of the censored data
        '''
        def _detection_limit_index(row):
            '''
            Helper function to create an array of indices for the
            detection  limits (self.cohn) corresponding to each
            data point
            '''
            det_limit_index = np.zeros(len(self.data[self.result_name]))
            if self.cohn.shape[0] > 0:
                index, = np.where(self.cohn['DL'] <= row[self.result_name])
                det_limit_index = index[-1]
            else:
                det_limit_index = 0

            return det_limit_index

        def _select_modeled(row):
            '''
            Helper fucntion to select "final" data from original detects
            and estimated non-detects
            '''
            if row[self.censorship_name]:
                return row['modeled_data']
            else:
                return row[self.result_name]

        def _select_half_detection_limit(row, fraction=0.5):
            '''
            Helper function to select half cohn when there are
            too few detections
            '''
            if row[self.censorship_name]:
                return fraction * row[self.result_name]
            else:
                return row[self.result_name]

        # create a det_limit_index column that references self.cohn
        self.data['det_limit_index'] = self.data.apply(_detection_limit_index, axis=1)

        # compute the ranks of the data
        self.data['rank'] = 1
        rank_columns = ['det_limit_index', self.censorship_name, 'rank']
        group_colums = ['det_limit_index', self.censorship_name]
        rankgroups = self.data[rank_columns].groupby(by=group_colums)
        self.data['rank'] = rankgroups.transform(lambda x: x.cumsum())

        # detect/non-detect selectors
        uncensored_mask = self.data[self.censorship_name] == False
        censored_mask = self.data[self.censorship_name] == True

        # if there are no non-detects, just spit everything back out
        if self.ncen == 0:
            self.data['modeled'] = self.data[self.result_name]
            self.data.sort(columns=[self.result_name], inplace=True)
            ppos, sorted_res = stats.probplot(
                self.data[self.result_name], fit=False
            )
            self.data['plot_pos'] = stats.norm.cdf(ppos)

        # if there are too few detects, use a fractoin of the detection limit
        elif (self.nobs - self.ncen < self.min_uncensored or
              self.ncen/self.nobs > self.max_fraction_censored):

            self.data['modeled'] = self.data.apply(
                _select_half_detection_limit, axis=1
            )

            ppos, sorted_res = stats.probplot(
                self.data[self.result_name], fit=False
            )
            self.data['plot_pos'] = stats.norm.cdf(ppos)

        # in most cases, actually use the MR method to estimate NDs
        else:
            # compute the PE values
            # (TODO: remove loop in place of apply)
            for j in self.cohn.index[:-1][::-1]:
                self.cohn.iloc[j]['prob_exceedance'] = (
                    self.cohn.iloc[j+1]['prob_exceedance'] +
                    self.cohn.iloc[j]['nuncen_above'] /
                   (
                        self.cohn.iloc[j]['nuncen_above'] +
                        self.cohn.iloc[j]['nobs_below']
                   ) * (1 - self.cohn.loc[j+1]['prob_exceedance'])
                )


            # compute the plotting position of the data (uses the PE stuff)
            self._compute_plotting_positions()

            # estimate a preliminary value of the Z-scores
            self.data['Zprelim'] = stats.norm.ppf(self.data['plot_pos'])

            # fit a line to the logs of the detected data
            self.fit_params = stats.linregress(
                self.data['Zprelim'][uncensored_mask],
                self.transform_in(self.data[self.result_name][uncensored_mask])
            )

            # pull out the slope and intercept for use later
            slope, intercept = self.fit_params[:2]

            # model the data based on the best-fit curve
            self.data['modeled_data'] = self.transform_out(
                slope * self.data['Zprelim'][censored_mask] + intercept
            )

            # select out the final data
            self.data['modeled'] = self.data.apply(_select_modeled, axis=1)

        # create the debug attribute as a copy of the self.data attribute
        self.debug = self.data.copy(deep=True)

        # select out only the necessary columns for data
        final_cols = ['modeled', self.result_name, self.censorship_name]
        self.data = self.data[final_cols]
        return self

    def plot(self, ax=None, show_raw=True, raw_kwds={}, model_kwds={},
             leg_kwds={}, ylog=True):
        '''
        Generate a QQ plot of the raw (censored) and modeled data.

        Parameters
        ----------
        ax : optional matplotlib Axes
            The axis on which the figure will be drawn. If no specified
            a new one is created.
        show_raw : optional boolean (default = True)
            Toggles on (True) or off (False) the drawing of the censored
            quantiles.
        raw_kwds : optional dict
            Plotting parameters for the censored data. Passed directly
            to `ax.plot`.
        model_kwds : optional dict
            Plotting parameters for the modeled data. Passed directly to
            `ax.plot`.
        leg_kwds : optional dict
            Optional kwargs for the legend, which is only drawn if
            `show_raw` is True. Passed directly to `ax.legend`.
        ylog : optional boolean (default = True)
            Toggles the logarthmic scale of the y-axis.

        Returns
        -------
        ax : matplotlib Axes

        '''

        fig, ax = utils.create_mpl_ax(ax)

        # legend options
        leg_params = {
            'loc': 'upper left',
            'fontsize': 8
        }
        leg_params.update(leg_kwds)

        # modeled data
        mod_symbols = {
            'marker': 'o',
            'markersize': 6,
            'markeredgewidth': 1.0,
            'markeredgecolor': 'blue',
            'markerfacecolor': 'blue',
            'linestyle': 'none',
            'label': 'Modeled data',
            'alpha': 0.87
        }
        mod_symbols.update(model_kwds)
        osm_mod, osr_mod = stats.probplot(
            self.data['modeled'], fit=False
        )
        ax.plot(osm_mod, osr_mod, **mod_symbols)

        # raw data
        if show_raw:
            raw_symbols = {
                'marker': 's',
                'markersize': 6,
                'markeredgewidth': 1.0,
                'markeredgecolor': '0.35',
                'markerfacecolor': 'none',
                'linestyle': 'none',
                'label': 'Censored data',
                'alpha': 0.70
            }
            raw_symbols.update(raw_kwds)
            osm_raw, osr_raw = stats.probplot(
                self.data[self.result_name], fit=False
            )
            ax.plot(osm_raw, osr_raw, **raw_symbols)
            ax.legend(**leg_params)

        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Observations')
        if ylog:
            ax.set_yscale('log')

        return ax
