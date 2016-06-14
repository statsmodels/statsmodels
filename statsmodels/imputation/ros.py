"""
Implementation of Regression on Order Statistics for imputing left-
censored (non-detect data)

Method described in *Nondetects and Data Analysis* by Dennis R.
Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)
values of a dataset.

Author: Paul M. Hobson
Company: Geosyntec Consultants (Portland, OR)
Date: 2016-06-14

"""

from __future__ import division
import warnings

import numpy
from scipy import stats
import pandas


def _ros_sort(df, result, censorship, warn=False):
    """
    This function prepares a dataframe for ROS.

    It sorts ascending with
    left-censored observations first. Censored results larger than the
    maximum uncensored results are removed from the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame

    result : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        result is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    ------
    sorted_df : pandas.DataFrame
        The sorted dataframe with all columns dropped except the
        result and censorship columns.

    """

    # separate uncensored data from censored data
    censored = df[df[censorship]].sort_values(by=result)
    uncensored = df[~df[censorship]].sort_values(by=result)

    if censored[result].max() > uncensored[result].max():
        censored = censored[censored[result] <= uncensored[result].max()]

        if warn:
            msg = ("Dropping censored results greater than "
                   "the max uncensored result.")
            warnings.warn(msg)

    return censored.append(uncensored)[[result, censorship]].reset_index(drop=True)


def cohn_numbers(df, result, censorship):
    """
    Computes the Cohn numbers for the detection limits in the dataset.

    The Cohn Numbers are:

        - :math:`A_j =` the number of uncensored obs above the jth
          threshold.
        - :math:`B_j =` the number of observations (cen & uncen) below
          the jth threshold.
        - :math:`C_j =` the number of censored observations at the jth
          threshold.
        - :math:`\mathrm{PE}_j =` the probability of exceeding the jth
          threshold
        - :math:`\mathrm{DL}_j =` the unique, sorted detection limits
        - :math:`\mathrm{DL}_{j+1} = \mathrm{DL}_j` shifted down a
          single index (row)

    Parameters
    ----------
    dataframe : pandas.DataFrame

    result : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        result is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    -------
    cohn : pandas.DataFrame

    """

    def nuncen_above(row):
        """ A, the number of uncensored obs above the given threshold.
        """

        # index of results above the lower_dl DL
        above = df[result] >= row['lower_dl']

        # index of results below the upper_dl DL
        below = df[result] < row['upper_dl']

        # index of non-detect results
        detect = df[censorship] == False

        # return the number of results where all conditions are True
        return df[above & below & detect].shape[0]

    def nobs_below(row):
        """ B, the number of observations (cen & uncen) below the given
        threshold
        """

        # index of data less than the lower_dl DL
        less_than = df[result] < row['lower_dl']

        # index of data less than or equal to the lower_dl DL
        less_thanequal = df[result] <= row['lower_dl']

        # index of detects, non-detects
        uncensored = df[censorship] == False
        censored = df[censorship] == True

        # number results less than or equal to lower_dl DL and non-detect
        LTE_censored = df[less_thanequal & censored].shape[0]

        # number of results less than lower_dl DL and detected
        LT_uncensored = df[less_than & uncensored].shape[0]

        # return the sum
        return LTE_censored + LT_uncensored

    def ncen_equal(row):
        """ C, the number of censored observations at the given
        threshold.
        """

        censored_index = df[censorship]
        censored_data = df[result][censored_index]
        censored_below = censored_data == row['lower_dl']
        return censored_below.sum()

    def set_upper_limit(cohn):
        """ Sets the upper_dl DL for each row of the Cohn dataframe. """
        if cohn.shape[0] > 1:
            return cohn['lower_dl'].shift(-1).fillna(value=numpy.inf)
        else:
            return [numpy.inf]

    def compute_PE(A, B):
        """ Computes the probability of excedance for each row of the
        Cohn dataframe. """
        N = len(A)
        PE = numpy.empty(N, dtype='float64')
        PE[-1] = 0.0
        for j in range(N-2, -1, -1):
            PE[j] = PE[j+1] + (1 - PE[j+1]) * A[j] / (A[j] + B[j])

        return PE

    # unique, sorted detection limts
    censored_data = df[censorship]
    DLs = pandas.unique(df.loc[censored_data, result])
    DLs.sort()

    # if there is a results smaller than the minimum detection limit,
    # add that value to the array
    if DLs.shape[0] > 0:
        if df[result].min() < DLs.min():
            DLs = numpy.hstack([df[result].min(), DLs])

        # create a dataframe
        cohn = (
            pandas.DataFrame(DLs, columns=['lower_dl'])
                .assign(upper_dl=lambda df: set_upper_limit(df))
                .assign(nuncen_above=lambda df: df.apply(nuncen_above, axis=1))
                .assign(nobs_below=lambda df: df.apply(nobs_below, axis=1))
                .assign(ncen_equal=lambda df: df.apply(ncen_equal, axis=1))
                .reindex(range(DLs.shape[0] + 1))
                .assign(prob_exceedance=lambda df: compute_PE(df['nuncen_above'], df['nobs_below']))
        )

    else:
        dl_cols = ['lower_dl', 'upper_dl', 'nuncen_above',
                   'nobs_below', 'ncen_equal', 'prob_exceedance']
        cohn = pandas.DataFrame(numpy.empty((0, len(dl_cols))), columns=dl_cols)

    return cohn


def _detection_limit_index(res, cohn):
    """
    Locates the corresponding detection limit for each observation.

    Basically, creates an array of indices for the detection limits
    (Cohn numbers) corresponding to each data point.

    Parameters
    ----------
    res : float
        A single observed result from the larger dataset.

    cohn : pandas.DataFrame
        Dataframe of Cohn numbers.

    Returns
    -------
    det_limit_index : int
        The index of the corresponding detection limit in `cohn`

    See also
    --------
    cohn_numbers

    """

    if cohn.shape[0] > 0:
        index, = numpy.where(cohn['lower_dl'] <= res)
        det_limit_index = index[-1]
    else:
        det_limit_index = 0

    return det_limit_index


def _ros_group_rank(df, dl_idx, censorship):
    """
    Ranks each result within the data groups.

    In this case, the groups are defined by the record's detection
    limit index and censorship status.

    Parameters
    ----------
    df : pandas.DataFrame

    dl_idx : str
        Name of the column in the dataframe the index of the result's
        corresponding detection limit in the `cohn` dataframe.

    censorship : str
        Name of the column in the dataframe that indicates that a
        result is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    -------
    ranks : numpy.array
        Array of ranks for the dataset.

    """

    ranks = (
        df.assign(rank=1)
          .groupby(by=[dl_idx, censorship])['rank']
          .transform(lambda g: g.cumsum())
    )
    return ranks


def _ros_plot_pos(row, censorship, cohn):
    """
    ROS-specific plotting positions.

    Computes the plotting position for an observation based on its rank,
    censorship status, and detection limit index.

    Parameters
    ----------
    row : pandas.Series or dict-like
        Full observation (row) from a censored dataset. Requires a
        'rank', 'detection_limit', and `censorship` column.

    censorship : str
        Name of the column in the dataframe that indicates that a
        result is left-censored. (i.e., True -> censored,
        False -> uncensored)

    cohn : pandas.DataFrame
        Dataframe of Cohn numbers.

    Returns
    -------
    plotting_position : float

    See also
    --------
    cohn_numbers

    """

    DL_index = row['det_limit_index']
    rank = row['rank']
    censored = row[censorship]

    dl_1 = cohn.iloc[DL_index]
    dl_2 = cohn.iloc[DL_index + 1]
    if censored:
        return (1 - dl_1['prob_exceedance']) * rank / (dl_1['ncen_equal']+1)
    else:
        return (1 - dl_1['prob_exceedance']) + (dl_1['prob_exceedance'] - dl_2['prob_exceedance']) * \
                rank / (dl_1['nuncen_above']+1)


def _norm_plot_pos(results):
    """
    Computes standard normal (Gaussian) plotting positions using scipy.

    Parameters
    ----------
    results : array-like
        Sequence of observed quantities.

    Returns
    -------
    plotting_position : array of floats

    """
    ppos, sorted_res = stats.probplot(results, fit=False)
    return stats.norm.cdf(ppos)


def plotting_positions(df, censorship, cohn):
    """
    Compute the plotting positions for the observations.

    The ROS-specific plotting postions are based on the observations'
    rank, censorship status, and corresponding detection limit.

    Parameters
    ----------
    df : pandas.DataFrame

    censorship : str
        Name of the column in the dataframe that indicates that a
        result is left-censored. (i.e., True -> censored,
        False -> uncensored)

    cohn : pandas.DataFrame
        Dataframe of Cohn numbers.

    Returns
    -------
    plotting_position : array of float

    See also
    --------
    cohn_numbers

    """

    plot_pos = df.apply(lambda r: _ros_plot_pos(r, censorship, cohn), axis=1)

    # correctly sort the plotting positions of the ND data:
    ND_plotpos = plot_pos[df[censorship]]
    ND_plotpos.values.sort()
    plot_pos.loc[df[censorship]] = ND_plotpos

    return plot_pos


def _impute(df, result, censorship, transform_in, transform_out):
    """
    Executes the basic regression on order stat (ROS) proceedure.

    Uses ROS to impute censored from the best-fit line of a
    probability plot of the uncensored values.

    Parameters
    ----------
    df : pandas.DataFrame
    result : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.
    censorship : str
        Name of the column in the dataframe that indicates that a
        result is left-censored. (i.e., True -> censored,
        False -> uncensored)
    transform_in, transform_out : callable
        Transformations to be applied to the data prior to fitting
        the line and after estimated values from that line. Typically,
        `numpy.log` and `numpy.exp` are used, respectively.

    Returns
    -------
    estimated : pandas.DataFrame
        A new dataframe with two new columns: "estimated" and "final".
        The "estimated" column contains of the values inferred from the
        best-fit line. The "final" column contains the estimated values
        only where the original results were censored, and the original
        results everwhere else.

    """

    # detect/non-detect selectors
    uncensored_mask = df[censorship] == False
    censored_mask = df[censorship] == True

    # fit a line to the logs of the detected data
    fit_params = stats.linregress(
        df['Zprelim'][uncensored_mask],
        transform_in(df[result][uncensored_mask])
    )

    # pull out the slope and intercept for use later
    slope, intercept = fit_params[:2]

    # model the data based on the best-fit curve
    df = (
        df.assign(estimated=transform_out(slope * df['Zprelim'][censored_mask] + intercept))
          .assign(final=lambda df: numpy.where(df[censorship], df['estimated'], df[result]))
    )

    return df


def _do_ros(df, result, censorship, transform_in, transform_out):
    """
    Dataframe-centric function to impute censored valies with ROS.

    Prepares a dataframe for, and then esimates the values of a censored
    dataset using Regression on Order Statistics

    Parameters
    ----------
    df : pandas.DataFrame

    result : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        result is left-censored. (i.e., True -> censored,
        False -> uncensored)

    transform_in, transform_out : callable
        Transformations to be applied to the data prior to fitting
        the line and after estimated values from that line. Typically,
        `numpy.log` and `numpy.exp` are used, respectively.

    Returns
    -------
    estimated : pandas.DataFrame
        A new dataframe with two new columns: "estimated" and "final".
        The "estimated" column contains of the values inferred from the
        best-fit line. The "final" column contains the estimated values
        only where the original results were censored, and the original
        results everwhere else.

    """

    # compute the Cohn numbers
    cohn = cohn_numbers(df, result=result, censorship=censorship)

    modeled = (
        df.pipe(_ros_sort, result=result, censorship=censorship)
          .assign(det_limit_index=lambda df: df[result].apply(_detection_limit_index, args=(cohn,)))
          .assign(rank=lambda df: _ros_group_rank(df, 'det_limit_index', censorship))
          .assign(plot_pos=lambda df: plotting_positions(df, censorship, cohn))
          .assign(Zprelim=lambda df: stats.norm.ppf(df['plot_pos']))
          .pipe(_impute, result, censorship, transform_in, transform_out)
    )

    return modeled


def impute_ros(result, censorship, df=None, min_uncensored=2,
           max_fraction_censored=0.8, substitution_fraction=0.5,
           transform_in=numpy.log, transform_out=numpy.exp,
           as_array=True):
    """
    Impute censored dataset using Regression on Order Statistics (ROS).

    Method described in *Nondetects and Data Analysis* by Dennis R.
    Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)
    values of a dataset. When there is insufficient non-censorded data,
    simple substitution is used.

    Parameters
    ----------
    result : str or array-like
        Label of the column or the float array of censored results

    censorship : str
        Label of the column or the bool array of the censorship
        status of the results.

          * True if censored,
          * False if uncensored

    df : pandas.DataFrame, optional
        If `result` and `censorship` are labels, this is the DataFrame
        that contains those columns.

    min_uncensored : int (default is 2)
        The minimum number of uncensored values required before ROS
        can be used to impute the censored results. When this criterion
        is not met, simple substituion is used instead.

    max_fraction_censored : float (default is 0.8)
        The maximum fraction of censored data below which ROS can be
        used to impute the censored results. When this fraction is
        exceeded, simple substituion is used instead.

    substitution_fraction : float (default is 0.5)
        The fraction of the detection limit to be used during simple
        substitution of the censored values.

    transform_in : callable (default is numpy.log)
        Transformation to be applied to the values prior to fitting a
        line to the plotting positions vs. uncensored values.

    transform_out : callable (default is numpy.exp)
        Transformation to be applied to the imputed censored values
        estimated from the previously computed best-fit line.

    as_array : bool (default is True)
        When True, a numpy array of the imputed results is returned.
        Otherwise, a modified copy of the original dataframe with all
        of the intermediate calculations is returned.

    Returns
    -------
    imputed : numpy.array (default) or pandas.DataFrame
        The final results where the censored values have either been
        imputed through ROS or substituted as a fraction of the
        detection limit.

    """

    # process arrays into a dataframe, if necessary
    if df is None:
        df = pandas.DataFrame({'res': result, 'cen': censorship})
        result = 'res'
        censorship = 'cen'

    # basic counts/metrics of the dataset
    N_observations = df.shape[0]
    N_censored = df[censorship].astype(int).sum()
    N_uncensored = N_observations - N_censored
    fraction_censored = N_censored / N_observations

    # add plotting positions if there are no censored values
    if N_censored == 0:
        output = df[[result, censorship]].assign(final=df[result])

    # substitute w/ fraction of the DLs if there's insufficient
    # uncensored data
    elif (N_uncensored < min_uncensored) or (fraction_censored > max_fraction_censored):
        final = numpy.where(df[censorship], df[result] * substitution_fraction, df[result])
        output = df.assign(final=final)[[result, censorship, 'final']]

    # normal ROS stuff
    else:
        output = _do_ros(df, result, censorship, transform_in, transform_out)

    # convert to an array if necessary
    if as_array:
        output = output['final'].values

    return output
