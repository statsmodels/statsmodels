"""
documentation comment
"""

import numpy as np
import statsmodels.base.model as base
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools import data as data_tools
from scipy.stats.distributions import norm
import pandas as pd
import patsy
from statsmodels.compat.collections import OrderedDict
from statsmodels.compat import range
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.base._penalties import Penalty
from statsmodels.compat.numpy import np_matrix_rank
import statsmodels.genmod.families as fams

from pandas import DataFrame

from statsmodels.tools.numdiff import (approx_fprime,
                                       approx_hess_cs)


def _get_exog_re_names(self, exog_re):
    """
    Passes through if given a list of names. Otherwise, gets pandas names
    or creates some generic variable names as needed.
    """
    if self.k_re == 0:
        return []
    if isinstance(exog_re, pd.DataFrame):
        return exog_re.columns.tolist()
    elif isinstance(exog_re, pd.Series) and exog_re.name is not None:
        return [exog_re.name]
    elif isinstance(exog_re, list):
        return exog_re
    return ["Z{0}".format(k + 1) for k in range(exog_re.shape[1])]


class MixedGLMParams(object):
    """
    This class represents a parameter state for a generalized mixed
    linear model.

    Parameters
    ----------
    k_fe : integer
        The number of covariates with fixed effects.
    k_re : integer
        The number of covariates with random effects.
    use_sqrt : boolean
        If True, the covariance matrix is stored using as the lower
        triangle of its Cholesky square root, otherwise it is stored
        as the lower triangle of the covariance matrix.
    Notes
    -----
    This object represents the parameter state for the model in which
    the scale parameter has been profiled out.
    """

    def __init__(self, k_fe, k_re, use_sqrt=True):

        self.k_fe = k_fe
        self.k_re = k_re
        self.k_re2 = k_re * (k_re + 1) // 2
        self.k_tot = self.k_fe + self.k_re2
        self.use_sqrt = use_sqrt
        self._ix = np.tril_indices(self.k_re)
        self._params = np.zeros(self.k_tot)

    def from_packed(params, k_fe, use_sqrt):
        """
        Create a MixedGLMParams object from packed parameter vector.

        Parameters
        ----------
        params : array-like
            The mode parameters packed into a single vector.
        k_fe : integer
            The number of covariates with fixed effects
        use_sqrt : boolean
            If True, the random effects covariance matrix is stored as
            its Cholesky factor, otherwise the lower trianle of the
            covariance matrix is stored.

        Returns
        -------
        A MixedGLMParams object.
        """

        k_re2 = len(params) - k_fe
        k_re = (-1 + np.sqrt(1 + 8 * k_re2)) / 2
        if k_re != int(k_re):
            raise ValueError('Length of `packed` '
                             '  not compatible with value of `fe`.')
        k_re = int(k_re)

        pa = MixedGLMParams(k_fe, k_re, use_sqrt)
        pa.set_packed(params)
        return pa

    from_packed = staticmethod(from_packed)

    def from_components(fe_params, cov_re=None, cov_re_sqrt=None,
                        use_sqrt=True):
        """
        Create a MixedGLMParams object from each parameter component.

        Parameters
        ----------
        fe_params : array-like
            The fixed effects parameter (a 1-dimensional array).
        cov_re : array-like
            The random effects covariance matrix (a square, symmetric
            2-dimensional array).
        cov_re_sqrt : array-like
            The Cholesky (lower triangular) square root of the random
            effects covariance matrix.
        use_sqrt : boolean
            If True, the random effects covariance matrix is stored as
            the lower triangle of its Cholesky factor, otherwise the
            lower triangle of the covariance matrix is stored.

        Returns
        -------
        A MixedGLMParams object.
        """

        k_fe = len(fe_params)
        k_re = cov_re.shape[0]
        pa = MixedGLMParams(k_fe, k_re, use_sqrt)
        pa.set_fe_params(fe_params)
        pa.set_cov_re(cov_re)

        return pa

    from_components = staticmethod(from_components)

    def copy(self):
        """
        Returns a copy of the object.
        """

        obj = MixedGLMParams(self.k_fe, self.k_re, self.use_sqrt)
        obj.set_packed(self.get_packed().copy())
        return obj

    def get_packed(self, use_sqrt=None):
        """
        Returns the model parameters packed into a single vector.

        Parameters
        ----------
        use_sqrt : None or bool
            If None, `use_sqrt` has the value of this instance's
            `use_sqrt`.  Otherwise it is set to the given value.
        """

        if (use_sqrt is None) or (use_sqrt == self.use_sqrt):
            return self._params

        pa = self._params.copy()
        cov_re = self.get_cov_re()

        if use_sqrt:
            L = np.linalg.cholesky(cov_re)
            pa[self.k_fe:] = L[self._ix]
        else:
            pa[self.k_fe:] = cov_re[self._ix]

        return pa

    def set_packed(self, params):
        """
        Sets the packed parameter vector to the given vector, without
        any validity checking.
        """
        self._params = params

    def get_fe_params(self):
        """
        Returns the fixed effects paramaters as a ndarray.
        """
        return self._params[0:self.k_fe]

    def set_fe_params(self, fe_params):
        """
        Set the fixed effect parameters to the given vector.
        """
        self._params[0:self.k_fe] = fe_params

    def set_cov_re(self, cov_re=None, cov_re_sqrt=None):
        """
        Set the random effects covariance matrix to the given value.

        Parameters
        ----------
        cov_re : array-like
            The random effects covariance matrix.
        cov_re_sqrt : array-like
            The Cholesky square root of the random effects covariance
            matrix.  Only the lower triangle is read.

        Notes
        -----
        The first of `cov_re` and `cov_re_sqrt` that is not None is
        used.
        """

        if cov_re is not None:
            if self.use_sqrt:
                cov_re_sqrt = np.linalg.cholesky(cov_re)
                self._params[self.k_fe:] = cov_re_sqrt[self._ix]
            else:
                self._params[self.k_fe:] = cov_re[self._ix]

        elif cov_re_sqrt is not None:
            if self.use_sqrt:
                self._params[self.k_fe:] = cov_re_sqrt[self._ix]
            else:
                cov_re = np.dot(cov_re_sqrt, cov_re_sqrt.T)
                self._params[self.k_fe:] = cov_re[self._ix]

    def get_cov_re(self):
        """
        Returns the random effects covariance matrix.
        """
        pa = self._params[self.k_fe:]

        cov_re = np.zeros((self.k_re, self.k_re))
        cov_re[self._ix] = pa
        if self.use_sqrt:
            cov_re = np.dot(cov_re, cov_re.T)
        else:
            cov_re = (cov_re + cov_re.T) - np.diag(np.diag(cov_re))

        return cov_re


def _smw_solver(s, A, AtA, B, BI):
    """
    Solves the system (s*I + A*B*A') * x = rhs for an arbitrary rhs.

    Parameters
    ----------
    s : scalar
        See above for usage
    A : square symmetric ndarray
        See above for usage
    AtA : square ndarray
        A.T * A
    B : square symmetric ndarray
        See above for usage
    BI : square symmetric ndarray
        The inverse of `B`.

    Returns
    -------
    A function that takes `rhs` as an input argument and returns a
    solution to the linear system defined above.
    """

    # Use SMW identity
    qmat = BI + AtA / s
    qmati = np.linalg.solve(qmat, A.T)

    def solver(rhs):
        ql = np.dot(qmati, rhs)
        ql = np.dot(A, ql)
        rslt = rhs / s - ql / s ** 2
        return rslt

    return solver


def _smw_logdet(s, A, AtA, B, BI, B_logdet):
    """
    Use the matrix determinant lemma to accelerate the calculation of
    the log determinant of s*I + A*B*A'.

    Parameters
    ----------
    s : scalar
        See above for usage
    A : square symmetric ndarray
        See above for usage
    AtA : square matrix
        A.T * A
    B : square symmetric ndarray
        See above for usage
    BI : square symmetric ndarray
        The inverse of `B`; can be None if B is singular.
    B_logdet : real
        The log determinant of B

    Returns
    -------
    The log determinant of s*I + A*B*A'.
    """

    p = A.shape[0]
    ld = p * np.log(s)
    qmat = BI + AtA / s
    _, ld1 = np.linalg.slogdet(qmat)
    return B_logdet + ld + ld1


class MixedGLM(base.LikelihoodModel):
    """
    An object specifying a linear mixed effects model.  Use the `fit`
    method to fit the model and obtain a results object.

    Parameters
    ----------
    endog : 1d array-like
        The dependent variable
    exog : 2d array-like
        A matrix of covariates used to determine the
        mean structure (the "fixed effects" covariates).
    groups : 1d array-like
        A vector of labels determining the groups -- data from
        different groups are independent
    exog_re : 2d array-like
        A matrix of covariates used to determine the variance and
        covariance structure (the "random effects" covariates).  If
        None, defaults to a random intercept for each group.
    exog_vc : dict-like
        A dicationary containing specifications of the variance
        component terms.  See below for details.
    use_sqrt : bool
        If True, optimization is carried out using the lower
        triangle of the square root of the random effects
        covariance matrix, otherwise it is carried out using the
        lower triangle of the random effects covariance matrix.
    missing : string
        The approach to missing data handling
    scale : float
        The scale parameter for the overdispersed exponential
        family.
    family : statsmodels.genmod.families.Family
        The GLM family. Defaults to Gaussian.

    Notes
    -----
    `exog_vc` is a dictionary of dictionaries.  Specifically,
    `exog_vc[a][g]` is a matrix whose columns are linearly combined
    using independent random coefficients.  This random term then
    contributes to the variance structure of the data for group `g`.
    The random coefficients all have mean zero, and have the same
    variance.  The matrix must be `m x k`, where `m` is the number of
    observations in group `g`.  The number of columns may differ among
    the top-level groups.

    The covariates in `exog`, `exog_re` and `exog_vc` may (but need
    not) partially or wholly overlap.

    `use_sqrt` should almost always be set to True.  The main use case
    for use_sqrt=False is when complicated patterns of fixed values in
    the covariance structure are set (using the `free` argument to
    `fit`) that cannot be expressed in terms of the Cholesky factor L.

    Examples
    --------
    A basic mixed model with fixed effects for the columns of
    ``exog`` and a random intercept for each distinct value of
    ``group``:

    >>> model = sm.MixedGLM(endog, exog, groups)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    correlated random coefficients for the columns of ``exog_re``:

    >>> model = sm.MixedGLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    independent random coefficients for the columns of ``exog_re``:

    >>> free = MixedGLMParams.from_components(fe_params=np.ones(exog.shape[1]),
                     cov_re=np.eye(exog_re.shape[1]))
    >>> model = sm.MixedGLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit(free=free)

    A different way to specify independent random coefficients for the
    columns of ``exog_re``.  In this example ``groups`` must be a
    Pandas Series with compatible indexing with ``exog_re``, and
    ``exog_re`` has two columns.

    >>> g = pd.groupby(groups, by=groups).groups
    >>> vc = {}
    >>> vc['1'] = {k : exog_re.loc[g[k], 0] for k in g}
    >>> vc['2'] = {k : exog_re.loc[g[k], 1] for k in g}
    >>> model = sm.MixedGLM(endog, exog, groups, vcomp=vc)
    >>> result = model.fit()
    """

    def __init__(self, endog, exog, groups, scale=1, family=fams.Gaussian(),
                 exog_re=None, exog_vc=None, use_sqrt=True, missing='none',
                 **kwargs):

        _allowed_kwargs = ["missing_idx", "design_info", "formula"]
        for x in kwargs.keys():
            if x not in _allowed_kwargs:
                raise ValueError("argument %s not permitted for "
                                 "MixedGLM initialization" % x)

        if (family is not None) and not \
                isinstance(family.link, tuple(family.safe_links)):
            import warnings
            warnings.warn("The %s link function does not respect "
                          "the domain of the %s family." %
                          (family.link.__class__.__name__,
                           family.__class__.__name__))

        self.family = family

        self.use_sqrt = use_sqrt
        self._scale = scale

        # Some defaults
        self.reml = True
        self.fe_pen = None
        self.re_pen = None

        # Needs to run early so that the names are sorted.
        self._setup_vcomp(exog_vc)

        # If there is one covariate, it may be passed in as a column
        # vector, convert these to 2d arrays.
        # TODO: Can this be moved up in the class hierarchy?
        #       yes, it should be done up the hierarchy
        if (exog is not None and
                data_tools._is_using_ndarray_type(exog, None) and
                exog.ndim == 1):
            exog = exog[:, None]
        if (exog_re is not None and
                data_tools._is_using_ndarray_type(exog_re, None) and
                exog_re.ndim == 1):
            exog_re = exog_re[:, None]

        # Calling super creates self.endog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(MixedGLM, self).__init__(endog, exog, groups=groups,
                                       exog_re=exog_re, missing=missing,
                                       **kwargs)

        self._init_keys.extend(["use_sqrt", "exog_vc"])

        self.k_fe = exog.shape[1]  # Number of fixed effects parameters

        if exog_re is None and exog_vc is None:
            # Default random effects structure (random intercepts).
            self.k_re = 1
            self.k_re2 = 1
            self.exog_re = np.ones((len(endog), 1), dtype=np.float64)
            self.data.exog_re = self.exog_re
            self.data.param_names = self.exog_names + ['Group RE']

        elif exog_re is not None:
            # Process exog_re the same way that exog is handled
            # upstream
            # TODO: this is wrong and should be handled upstream wholly
            self.data.exog_re = exog_re
            self.exog_re = np.asarray(exog_re)
            if self.exog_re.ndim == 1:
                self.exog_re = self.exog_re[:, None]
            # Model dimensions
            # Number of random effect covariates
            self.k_re = self.exog_re.shape[1]
            # Number of covariance parameters
            self.k_re2 = self.k_re * (self.k_re + 1) // 2

        else:
            # All random effects are variance components
            self.k_re = 0
            self.k_re2 = 0

        if not self.data._param_names:
            # HACK: could've been set in from_formula already
            # needs refactor
            (param_names, exog_re_names,
             exog_re_names_full) = self._make_param_names(exog_re)
            self.data.param_names = param_names
            self.data.exog_re_names = exog_re_names
            self.data.exog_re_names_full = exog_re_names_full

        self.k_params = self.k_fe + self.k_re2

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the groups.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        for i, g in enumerate(groups):
            row_indices[g].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.n_groups = len(self.group_labels)

        # Split the data by groups
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)

        # Precompute this.
        if self.exog_re is None:
            self.exog_re2_li = None
        else:
            self.exog_re2_li = [np.dot(x.T, x) for x in self.exog_re_li]

        # The total number of observations, summed over all groups
        self.n_totobs = sum([len(y) for y in self.endog_li])
        # why do it like the above?
        self.nobs = len(self.endog)

        # Set the fixed effects parameter names
        if self.exog_names is None:
            self.exog_names = ["FE%d" % (k + 1) for k in
                               range(self.exog.shape[1])]

        # Precompute this
        self._aex_r = []
        self._aex_r2 = []
        for i in range(self.n_groups):
            a, b = self._augment_exog(i)
            self._aex_r.append(a)
            self._aex_r2.append(b)

        # Precompute this
        self._lin, self._quad = self._reparam()

    def _setup_vcomp(self, exog_vc):
        if exog_vc is None:
            exog_vc = {}
        self.exog_vc = exog_vc
        self.k_vc = len(exog_vc)
        vc_names = list(set(exog_vc.keys()))
        vc_names.sort()
        self._vc_names = vc_names

    def _make_param_names(self, exog_re):
        """
        Returns the full parameter names list, just the exogenous random
        effects variables, and the exogenous random effects variables with
        the interaction terms.
        """
        exog_names = list(self.exog_names)
        exog_re_names = _get_exog_re_names(self, exog_re)
        param_names = []

        jj = self.k_fe
        for i in range(len(exog_re_names)):
            for j in range(i + 1):
                if i == j:
                    param_names.append(exog_re_names[i] + " RE")
                else:
                    param_names.append(exog_re_names[j] + " RE x " +
                                       exog_re_names[i] + " RE")
                jj += 1

        vc_names = [x + " RE" for x in self._vc_names]

        return exog_names + param_names + vc_names, exog_re_names, param_names

    @classmethod
    def from_formula(cls, formula, data, re_formula=None, vc_formula=None,
                     subset=None, family=fams.Gaussian(), scale=1,
                     *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array-like
            The data for the model. See Notes.
        re_formula : string
            A one-sided formula defining the variance structure of the
            model.  The default gives a random intercept for each
            group.
        vc_formula : dict-like
            Formulas describing variance components.  `vc_formula[vc]` is
            the formula for the component with variance parameter named
            `vc`.  The formula is processed into a matrix, and the columns
            of this matrix are linearly combined with independent random
            coefficients having mean zero and a common variance.
        subset : array-like
            An array-like object of booleans, integers, or index
            values that indicate the subset of df to use in the
            model. Assumes df is a `pandas.DataFrame`
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : Model instance

        Notes
        ------
        `data` must define __getitem__ with the keys in the formula
        terms args and kwargs are passed on to the model
        instantiation. E.g., a numpy structured or rec array, a
        dictionary, or a pandas DataFrame.

        If the variance component is intended to produce random
        intercepts for disjoint subsets of a group, specified by
        string labels or a categorical data value, always use '0 +' in
        the formula so that no overall intercept is included.

        If the variance components specify random slopes and you do
        not also want a random group-level intercept in the model,
        then use '0 +' in the formula to exclude the intercept.

        The variance components formulas are processed separately for
        each group.  If a variable is categorical the results will not
        be affected by whether the group labels are distinct or
        re-used over the top-level groups.

        This method currently does not correctly handle missing
        values, so missing values should be explicitly dropped from
        the DataFrame before calling this method.

        Examples
        --------
        Suppose we have an educational data set with students nested
        in classrooms nested in schools.  The students take a test,
        and we want to relate the test scores to the students' ages,
        while accounting for the effects of classrooms and schools.
        The school will be the top-level group, and the classroom is a
        nested group that is specified as a variance component.  Note
        that the schools may have different number of classrooms, and
        the classroom labels may (but need not be) different across
        the schools.

        >>> vc = {'classroom': '0 + C(classroom)'}
        >>> MixedGLM.from_formula('test_score ~ age', vc_formula=vc,
                                  re_formula='1', groups='school', data=data)

        Now suppose we also have a previous test score called
        'pretest'.  If we want the relationship between pretest
        scores and the current test to vary by classroom, we can
        specify a random slope for the pretest score

        >>> vc = {'classroom': '0 + C(classroom)', 'pretest': '0 + pretest'}
        >>> MixedGLM.from_formula('test_score ~ age + pretest', vc_formula=vc,
                                  re_formula='1', groups='school', data=data)

        The following model is almost equivalent to the previous one,
        but here the classroom random intercept and pretest slope may
        be correlated.

        >>> vc = {'classroom': '0 + C(classroom)'}
        >>> MixedGLM.from_formula('test_score ~ age + pretest', vc_formula=vc,
                                  re_formula='1 + pretest', groups='school',
                                  data=data)
        """

        if "groups" not in kwargs.keys():
            raise AttributeError("'groups' is a required keyword "
                                 "argument in MixedGLM.from_formula")

        if (family is not None) and not \
                isinstance(family.link, tuple(family.safe_links)):
            import warnings
            warnings.warn("The %s link function does not "
                          "respect the domain of the %s family." %
                          (family.link.__class__.__name__,
                           family.__class__.__name__))

        # If `groups` is a variable name, retrieve the data for the
        # groups variable.
        group_name = "Group"
        if type(kwargs["groups"]) == str:
            group_name = kwargs["groups"]
            kwargs["groups"] = np.asarray(data[kwargs["groups"]])

        if re_formula is not None:
            if re_formula.strip() == "1":
                # Work around Patsy bug, fixed by 0.3.
                exog_re = np.ones((data.shape[0], 1))
                exog_re_names = ["Group"]
            else:
                eval_env = kwargs.get('eval_env', None)
                if eval_env is None:
                    eval_env = 1
                elif eval_env == -1:
                    from patsy import EvalEnvironment

                    eval_env = EvalEnvironment({})
                exog_re = patsy.dmatrix(re_formula, data, eval_env=eval_env)
                exog_re_names = exog_re.design_info.column_names
                exog_re = np.asarray(exog_re)
            if exog_re.ndim == 1:
                exog_re = exog_re[:, None]
        else:
            exog_re = None
            if vc_formula is None:
                exog_re_names = ["groups"]
            else:
                exog_re_names = []

        if vc_formula is not None:
            eval_env = kwargs.get('eval_env', None)
            if eval_env is None:
                eval_env = 1
            elif eval_env == -1:
                from patsy import EvalEnvironment

                eval_env = EvalEnvironment({})

            exog_vc = {}
            data["_group"] = kwargs["groups"]
            gb = data.groupby("_group")
            kylist = list(gb.groups.keys())
            kylist.sort()
            for vc_name in vc_formula.keys():
                exog_vc[vc_name] = {}
                for group_ix, group in enumerate(kylist):
                    ii = gb.groups[group]
                    vcg = vc_formula[vc_name]
                    mat = patsy.dmatrix(vcg, data.loc[ii, :],
                                        eval_env=eval_env,
                                        return_type='dataframe')
                    exog_vc[vc_name][group] = np.asarray(mat)
            exog_vc = exog_vc
        else:
            exog_vc = None

        mod = super(MixedGLM, cls).from_formula(formula, data,
                                                subset=None,
                                                exog_re=exog_re,
                                                exog_vc=exog_vc,
                                                *args, **kwargs)

        # expand re names to account for pairs of RE
        (param_names,
         exog_re_names,
         exog_re_names_full) = mod._make_param_names(exog_re_names)

        mod.data.param_names = param_names
        mod.data.exog_re_names = exog_re_names
        mod.data.exog_re_names_full = exog_re_names_full
        mod.data.vcomp_names = mod._vc_names

        mod.family = family
        mod._scale = scale

        return mod

    def predict(self, params, exog=None):
        """
        Return predicted values from a design matrix.

        Parameters
        ----------
        params : array-like
            Parameters of a mixed linear model.  Can be either a
            MixedGLMParams instance, or a vector containing the packed
            model parameters in which the fixed effects parameters are
            at the beginning of the vector, or a vector containing
            only the fixed effects parameters.
        exog : array-like, optional
            Design / exogenous data for the fixed effects. Model exog
            is used if None.

        Returns
        -------
        An array of fitted values.  Note that these predicted values
        only reflect the fixed effects mean structure of the model.
        """
        if exog is None:
            exog = self.exog

        if isinstance(params, MixedGLMParams):
            params = params.fe_params
        else:
            params = params[0:self.k_fe]

        return np.dot(exog, params)

    def group_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """

        if array is None:
            return None

        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :])
                    for k in self.group_labels]

    def _reparam(self):
        """
        Returns parameters of the map converting parameters from the
        form used in optimization to the form returned to the user.

        Returns
        -------
        lin : list-like
            Linear terms of the map
        quad : list-like
            Quadratic terms of the map

        Notes
        -----
        If P are the standard form parameters and R are the
        transformed parameters (i.e. with the Cholesky square root
        covariance and square root transformed variane components),
        then P[i] = lin[i] * R + R' * quad[i] * R
        """

        k_fe, k_re, k_re2, k_vc = self.k_fe, self.k_re, self.k_re2, self.k_vc
        k_tot = k_fe + k_re2 + k_vc
        ix = np.tril_indices(self.k_re)

        lin = []
        for k in range(k_fe):
            e = np.zeros(k_tot)
            e[k] = 1
            lin.append(e)
        for k in range(k_re2):
            lin.append(np.zeros(k_tot))
        for k in range(k_vc):
            lin.append(np.zeros(k_tot))

        quad = []
        # Quadratic terms for fixed effects.
        for k in range(k_tot):
            quad.append(np.zeros((k_tot, k_tot)))

        # Quadratic terms for random effects covariance.
        ii = np.tril_indices(k_re)
        ix = [(a, b) for a, b in zip(ii[0], ii[1])]
        for i1 in range(k_re2):
            for i2 in range(k_re2):
                ix1 = ix[i1]
                ix2 = ix[i2]
                if (ix1[1] == ix2[1]) and (ix1[0] <= ix2[0]):
                    ii = (ix2[0], ix1[0])
                    k = ix.index(ii)
                    quad[k_fe + k][k_fe + i2, k_fe + i1] += 1
        for k in range(k_tot):
            quad[k] = 0.5 * (quad[k] + quad[k].T)

        # Quadratic terms for variance components.
        km = k_fe + k_re2
        for k in range(km, km + k_vc):
            quad[k][k, k] = 1

        return lin, quad

    def _augment_cov_re(self, cov_re, cov_re_inv, vcomp, group):
        """
        Returns the covariance matrix and its inverse for all random
        effects.  Also returns the adjustment to the determinant for
        this group.
        """
        if self.k_vc == 0:
            return cov_re, cov_re_inv, 0.

        vc_var = []
        for j, k in enumerate(self._vc_names):
            if group not in self.exog_vc[k]:
                continue
            vc_var.append([vcomp[j]] * self.exog_vc[k][group].shape[1])
        vc_var = np.concatenate(vc_var)

        m = cov_re.shape[0] + len(vc_var)
        cov_aug = np.zeros((m, m))
        cov_aug[0:self.k_re, 0:self.k_re] = cov_re
        ix = np.arange(self.k_re, m)
        cov_aug[ix, ix] = vc_var

        cov_aug_inv = np.zeros((m, m))
        cov_aug_inv[0:self.k_re, 0:self.k_re] = cov_re_inv
        cov_aug_inv[ix, ix] = 1 / vc_var

        return cov_aug, cov_aug_inv, np.sum(np.log(vc_var))

    def _augment_exog(self, group_ix):
        """
        Concatenate the columns for variance components to the columns
        for other random effects to obtain a single random effects
        exog matrix.  Returns the matrix and its cross product matrix.
        """
        ex_r = self.exog_re_li[group_ix] if self.k_re > 0 else None
        ex2_r = self.exog_re2_li[group_ix] if self.k_re > 0 else None
        if self.k_vc == 0:
            return ex_r, ex2_r

        group = self.group_labels[group_ix]
        ex = [ex_r] if self.k_re > 0 else []
        for j, k in enumerate(self._vc_names):
            if group not in self.exog_vc[k]:
                continue
            ex.append(self.exog_vc[k][group])
        ex = np.concatenate(ex, axis=1)

        ex2 = np.dot(ex.T, ex)
        return ex, ex2

    def loglike(self, params):
        """
        Evaluate the log-likelihood of the generalized linear mixed
        model.

        Parameters
        ----------
        params : MixedGLMParams, or array-like.
            The parameter values.

        Returns
        -------
        float
            The log-likelihood value at `params`.
        """

        if type(params) is not MixedGLMParams:
            params = MixedGLMParams.from_packed(params, self.k_fe,
                                                self.use_sqrt)

        fe_params = params.get_fe_params()
        cov_re = params.get_cov_re()

        _, cov_re_logdet = np.linalg.slogdet(cov_re)

        # Log of the integral over random effects
        # includes everything but the multi. normal consts
        ival = self._laplace(fe_params, cov_re, self._scale)

        # Multi. normal consts from the random effects
        nconst = - self.n_groups * self.k_re * np.log(2 * np.pi) / 2.0
        nconst -= self.n_groups * cov_re_logdet / 2.0

        likeval = nconst + ival

        return likeval

    def _laplace(self, fe_params, cov_re, scale, omethod='BFGS'):
        """
        Log of Laplace's approximation of integral over random effects of joint
        likelihood.

        The Laplace approximation is used to approximate exponential integrals.
        Here we use it to approximate the marginal log-likelihood by
        approximating the integral of the joint likelihood over the integral
        over the random effects.

        Parameters
        ----------
        fe_params : array-like, 1d
            The fixed effects parameters.
        cov_re : array-like, 2d
            The covariance matrix of the random effects.
        scale : float
            The overdispersion scale parameter for the exponential family.
        omethod : string
            The optimization method for scipy.optimize.minimize.

        Returns
        -------
        float
            The kernel of the marginal log-likelihood. We marginalize out the
            random effects. This is the kernel as it ignores the normalizing
            constants of the random effects' distribution.
        """

        # The max of the function in the exponent of our integral
        mp = self._get_map(fe_params, cov_re, scale, omethod)

        f, h = self._joint_like_hess(mp, fe_params, cov_re, scale)

        d = len(mp)

        # This is the log of the Laplace approximation.
        # Algebraically we take the log of integral's approximation to avoid
        # problems with numpy overflow with expressions like np.log(np.exp(.))
        ival = - f + (d / 2.0) * np.log(2 * np.pi) - (1 / 2.0) * h

        return(ival)

    def _gen_joint_like_score(self, fe_params, cov_re, scale):
        """
        Function and gradient of the joint log likelihood.

        The log-likelihood is for the data and random effects, viewed as a
        function of the random effects.

        Parameters
        ----------
        params : array-like, 1d
            The fixed effects parameters.
        cov_re : array-like, 2d
            The covariance matrix of the random effects.
        scale : float
            The overdispersion scale parameter for the exponential family.

        Returns
        -------
        function
            A function that takes a state for the random effects
            (vectorized) and returns the value of the (kernel of the) negative
            joint log likelihood and its gradient evaluated at the given state
            respectively in a tuple.
        """

        lin_pred = np.dot(self.exog, fe_params)
        lin_pred = self.group_list(lin_pred)

        def fun(ref):

            # Matrix of current state of random effects
            ref = np.reshape(ref, (self.n_groups, self.k_re))

            s = np.linalg.solve(cov_re, ref.T)
            f = -(ref.T * s).sum() / 2  # -1/2 * x'cov^(-1)x

            d = np.zeros((self.n_groups, self.k_re))

            # Build up log-likelihood group by group
            for k, g in enumerate(self.group_labels):
                lin_predr = lin_pred[k] + np.dot(self.exog_re_li[k], ref[k, :])
                mean = self.family.fitted(lin_predr)  # mu_i = h(eta_i)
                log_likes = self.family.loglike(self.endog_li[k], mean,
                                                scale=scale)
                f += log_likes

                d[k, :] = ((self.endog_li[k] - mean)[:, None] *
                           self.exog_re_li[k] * scale).sum(0)
                d[k, :] -= s[:, k]

            # We need negatives because scipy.optimize can only
            # find minima not maxima
            return -f, -d.ravel()

        return fun

    def _joint_like_hess(self, ref, fe_params, cov_re, scale):
        """
        Function and Hessian of the joint log likelihood evaluated at
        the given state.

        The log-likelihood is for the data and random effects, viewed as a
        function of the random effects.

        Parameters
        ----------
        ref : array-like
            The random effects.
        fe_params : array-like, 1d
            The fixed effects parameters.
        cov_re : array-like, 2d
            The covariance matrix of the random effects.
        scale : float
            The overdispersion scale parameter for the exponential family.

        Returns
        -------
        tuple
            A tuple containing two elements.
            First, the (kernel of the) negative joint log likelihood and second
            the log of the determinant of its Hessian evaluated at the given
            state.
        """

        lin_pred = np.dot(self.exog, fe_params)
        lin_pred = self.group_list(lin_pred)

        ref = np.reshape(ref, (self.n_groups, self.k_re))

        s = np.linalg.solve(cov_re, ref.T)
        f = -(ref.T * s).sum() / 2

        d2 = np.zeros(self.n_groups)

        for k, g in enumerate(self.group_labels):
            lin_predr = lin_pred[k] + np.dot(self.exog_re_li[k], ref[k, :])
            mean = self.family.fitted(lin_predr)
            f += self.family.loglike(self.endog_li[k], mean, scale=scale)
            va = self.family.variance(mean)
            hmat = va[:, None] * self.exog_re_li[k]
            hmat = np.dot(self.exog_re_li[k].T, hmat)
            hmat += np.linalg.inv(cov_re)
            _, d2[k] = np.linalg.slogdet(hmat)

        return -f, d2.sum()

    def _get_map(self, fe_params, cov_re, scale, omethod):
        """
        Obtain the MAP predictor of the random effects.

        The MAP (maximum a posteriori) predictor is the mode of the joint
        likelihood of the data and the random effects, viewed as a
        function of the random effects.

        Parameters
        ----------
        params : array-like, 1d
            The fixed effects parameters
        cov_re : array-like, 2d
            The covariance matrix of the random effects
        scale : float
            The overdispersion scale parameter for the exponential family.
        omethod : string
            The optimization method for scipy.optimize.minimize.

        Returns
        -------
        array-like, 1d
            The MAP predictor of the random effects.
        """

        fun = self._gen_joint_like_score(fe_params, cov_re, scale)
        x0 = np.random.normal(size=self.n_groups * self.k_re)

        result = minimize(fun, x0, jac=True, method=omethod)

        if not result.success:
            raise Warning(result.message)

        mp = np.reshape(result.x, (self.n_groups, self.k_re))
        return mp

    def _gen_dV_dPar(self, ex_r, solver, group, max_ix=None):
        """
        A generator that yields the element-wise derivative of the
        marginal covariance matrix with respect to the random effects
        variance and covariance parameters.

        ex_r : array-like
            The random effects design matrix
        solver : function
            A function that given x returns V^{-1}x, where V
            is the group's marginal covariance matrix.
        group : scalar
            The group label
        max_ix : integer or None
            If not None, the generator ends when this index
            is reached.
        """

        axr = solver(ex_r)

        # Regular random effects
        jj = 0
        for j1 in range(self.k_re):
            for j2 in range(j1 + 1):
                if max_ix is not None and jj > max_ix:
                    return
                # Need 2d
                mat_l, mat_r = ex_r[:, j1:j1 + 1], ex_r[:, j2:j2 + 1]
                vsl, vsr = axr[:, j1:j1 + 1], axr[:, j2:j2 + 1]
                yield jj, mat_l, mat_r, vsl, vsr, j1 == j2
                jj += 1

        # Variance components
        for ky in self._vc_names:
            if group in self.exog_vc[ky]:
                if max_ix is not None and jj > max_ix:
                    return
                mat = self.exog_vc[ky][group]
                axmat = solver(mat)
                yield jj, mat, mat, axmat, axmat, True
                jj += 1

    def score(self, params, profile_fe=True):
        """
        Returns the score vector of the profile log-likelihood.

        Notes
        -----
        The score vector that is returned is computed with respect to
        the parameterization defined by this model instance's
        `use_sqrt` attribute.
        """

        if type(params) is not MixedGLMParams:
            params = MixedGLMParams.from_packed(params, self.k_fe,
                                                self.k_re, self.use_sqrt)

        if profile_fe:
            params.fe_params = self.get_fe_params(params.cov_re, params.vcomp)

        if self.use_sqrt:
            score_fe, score_re, score_vc = \
                self.score_sqrt(params, calc_fe=not profile_fe)
        else:
            score_fe, score_re, score_vc = \
                self.score_full(params, calc_fe=not profile_fe)

        if self._freepat is not None:
            score_fe *= self._freepat.fe_params
            score_re *= self._freepat.cov_re[self._freepat._ix]
            score_vc *= self._freepat.vcomp

        if profile_fe:
            return np.concatenate((score_re, score_vc))
        else:
            return np.concatenate((score_fe, score_re, score_vc))

    def score_full(self, params, calc_fe):
        """
        Returns the score with respect to untransformed parameters.

        Calculates the score vector for the profiled log-likelihood of
        the mixed effects model with respect to the parameterization
        in which the random effects covariance matrix is represented
        in its full form (not using the Cholesky factor).

        Parameters
        ----------
        params : MixedGLMParams or array-like
            The parameter at which the score function is evaluated.
            If array-like, must contain the packed random effects
            parameters (cov_re and vcomp) without fe_params.
        calc_fe : boolean
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array-like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array-like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array-like
            The score vector with respect to variance components
            parameters.

        Notes
        -----
        `score_re` is taken with respect to the parameterization in
        which `cov_re` is represented through its lower triangle
        (without taking the Cholesky square root).
        """

        fe_params = params.fe_params
        cov_re = params.cov_re
        vcomp = params.vcomp

        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        score_fe = np.zeros(self.k_fe)
        score_re = np.zeros(self.k_re2)
        score_vc = np.zeros(self.k_vc)

        # Handle the covariance penalty.
        if self.cov_pen is not None:
            score_re -= self.cov_pen.grad(cov_re, cov_re_inv)

        # Handle the fixed effects penalty.
        if calc_fe and (self.fe_pen is not None):
            score_fe -= self.fe_pen.grad(fe_params)

        # resid' V^{-1} resid, summed over the groups (a scalar)
        rvir = 0.

        # exog' V^{-1} resid, summed over the groups (a k_fe
        # dimensional vector)
        xtvir = 0.

        # exog' V^{_1} exog, summed over the groups (a k_fe x k_fe
        # matrix)
        xtvix = 0.

        # V^{-1} exog' dV/dQ_jj exog V^{-1}, where Q_jj is the jj^th
        # covariance parameter.
        xtax = [0., ] * (self.k_re2 + self.k_vc)

        # Temporary related to the gradient of log |V|
        dlv = np.zeros(self.k_re2 + self.k_vc)

        # resid' V^{-1} dV/dQ_jj V^{-1} resid (a scalar)
        rvavr = np.zeros(self.k_re2 + self.k_vc)

        for group_ix, group in enumerate(self.group_labels):

            cov_aug, cov_aug_inv, _ = \
                self._augment_cov_re(cov_re, cov_re_inv, vcomp, group)

            exog = self.exog_li[group_ix]
            ex_r, ex2_r = self._aex_r[group_ix], self._aex_r2[group_ix]
            solver = _smw_solver(1., ex_r, ex2_r, cov_aug, cov_aug_inv)

            # The residuals
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval

            if self.reml:
                viexog = solver(exog)
                xtvix += np.dot(exog.T, viexog)

            # Contributions to the covariance parameter gradient
            vex = solver(ex_r)
            vir = solver(resid)
            for jj, matl, matr, vsl, vsr, sym in \
                    self._gen_dV_dPar(ex_r, solver, group):
                dlv[jj] = np.sum(matr * vsl)  # trace dot
                if not sym:
                    dlv[jj] += np.sum(matl * vsr)  # trace dot

                ul = np.dot(vir, matl)
                ur = ul.T if sym else np.dot(matr.T, vir)
                ulr = np.dot(ul, ur)
                rvavr[jj] += ulr
                if not sym:
                    rvavr[jj] += ulr.T

                if self.reml:
                    ul = np.dot(viexog.T, matl)
                    ur = ul.T if sym else np.dot(matr.T, viexog)
                    ulr = np.dot(ul, ur)
                    xtax[jj] += ulr
                    if not sym:
                        xtax[jj] += ulr.T

            # Contribution of log|V| to the covariance parameter
            # gradient.
            if self.k_re > 0:
                score_re -= 0.5 * dlv[0:self.k_re2]
            if self.k_vc > 0:
                score_vc -= 0.5 * dlv[self.k_re2:]

            rvir += np.dot(resid, vir)

            if calc_fe:
                xtvir += np.dot(exog.T, vir)

        fac = self.n_totobs
        if self.reml:
            fac -= self.k_fe

        if calc_fe and self.k_fe > 0:
            score_fe += fac * xtvir / rvir

        if self.k_re > 0:
            score_re += 0.5 * fac * rvavr[0:self.k_re2] / rvir
        if self.k_vc > 0:
            score_vc += 0.5 * fac * rvavr[self.k_re2:] / rvir

        if self.reml:
            xtvixi = np.linalg.inv(xtvix)
            for j in range(self.k_re2):
                score_re[j] += 0.5 * np.sum(xtvixi.T * xtax[j])  # trace dot
            for j in range(self.k_vc):
                score_vc[j] += 0.5 * np.sum(xtvixi.T * xtax[self.k_re2 + j])

        return score_fe, score_re, score_vc

    def score_sqrt(self, params, calc_fe=True):
        """
        Returns the score with respect to transformed parameters.

        Calculates the score vector with respect to the
        parameterization in which the random effects covariance matrix
        is represented through its Cholesky square root.

        Parameters
        ----------
        params : MixedGLMParams or array-like
            The model parameters.  If array-like must contain packed
            parameters that are compatible with this model instance.
        calc_fe : boolean
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array-like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array-like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array-like
            The score vector with respect to variance components
            parameters.
        """

        score_fe, score_re, score_vc = self.score_full(params, calc_fe=calc_fe)
        params_vec = params.get_packed(use_sqrt=True)

        score_full = np.concatenate((score_fe, score_re, score_vc))
        scr = 0.
        for i in range(len(params_vec)):
            v = self._lin[i] + 2 * np.dot(self._quad[i], params_vec)
            scr += score_full[i] * v
        score_fe = scr[0:self.k_fe]
        score_re = scr[self.k_fe:self.k_fe + self.k_re2]
        score_vc = scr[self.k_fe + self.k_re2:]

        return score_fe, score_re, score_vc

    def hessian(self, params):
        """
        Returns the model's Hessian matrix.

        Calculates the Hessian matrix for the linear mixed effects
        model with respect to the parameterization in which the
        covariance matrix is represented directly (without square-root
        transformation).

        Parameters
        ----------
        params : MixedGLMParams or array-like
            The model parameters at which the Hessian is calculated.
            If array-like, must contain the packed parameters in a
            form that is compatible with this model instance.

        Returns
        -------
        hess : 2d ndarray
            The Hessian matrix, evaluated at `params`.
        """

        if type(params) is not MixedGLMParams:
            params = MixedGLMParams.from_packed(params, self.k_fe, self.k_re,
                                                use_sqrt=self.use_sqrt)

        fe_params = params.fe_params
        vcomp = params.vcomp
        cov_re = params.cov_re
        if self.k_re > 0:
            cov_re_inv = np.linalg.inv(cov_re)
        else:
            cov_re_inv = np.empty((0, 0))

        # Blocks for the fixed and random effects parameters.
        hess_fe = 0.
        hess_re = np.zeros((self.k_re2 + self.k_vc, self.k_re2 + self.k_vc))
        hess_fere = np.zeros((self.k_re2 + self.k_vc, self.k_fe))

        fac = self.n_totobs
        if self.reml:
            fac -= self.exog.shape[1]

        rvir = 0.
        xtvix = 0.
        xtax = [0., ] * (self.k_re2 + self.k_vc)
        m = self.k_re2 + self.k_vc
        B = np.zeros(m)
        D = np.zeros((m, m))
        F = [[0.] * m for k in range(m)]
        for k, group in enumerate(self.group_labels):

            cov_aug, cov_aug_inv, _ = \
                self._augment_cov_re(cov_re, cov_re_inv, vcomp, group)

            exog = self.exog_li[k]
            ex_r, ex2_r = self._aex_r[k], self._aex_r2[k]
            solver = _smw_solver(1., ex_r, ex2_r, cov_aug, cov_aug_inv)

            # The residuals
            resid = self.endog_li[k]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval

            viexog = solver(exog)
            xtvix += np.dot(exog.T, viexog)
            vir = solver(resid)
            rvir += np.dot(resid, vir)

            for jj1, matl1, matr1, vsl1, vsr1, sym1 in \
                    self._gen_dV_dPar(ex_r, solver, group):

                ul = np.dot(viexog.T, matl1)
                ur = np.dot(matr1.T, vir)
                hess_fere[jj1, :] += np.dot(ul, ur)
                if not sym1:
                    ul = np.dot(viexog.T, matr1)
                    ur = np.dot(matl1.T, vir)
                    hess_fere[jj1, :] += np.dot(ul, ur)

                if self.reml:
                    ul = np.dot(viexog.T, matl1)
                    ur = ul if sym1 else np.dot(viexog.T, matr1)
                    ulr = np.dot(ul, ur.T)
                    xtax[jj1] += ulr
                    if not sym1:
                        xtax[jj1] += ulr.T

                ul = np.dot(vir, matl1)
                ur = ul if sym1 else np.dot(vir, matr1)
                B[jj1] += np.dot(ul, ur) * (1 if sym1 else 2)

                # V^{-1} * dV/d_theta
                E = np.dot(vsl1, matr1.T)
                if not sym1:
                    E += np.dot(vsr1, matl1.T)

                for jj2, matl2, matr2, vsl2, vsr2, sym2 in \
                        self._gen_dV_dPar(ex_r, solver, group, jj1):

                    re = np.dot(matr2.T, E)
                    rev = np.dot(re, vir[:, None])
                    vl = np.dot(vir[None, :], matl2)
                    vt = 2 * np.dot(vl, rev)

                    if not sym2:
                        le = np.dot(matl2.T, E)
                        lev = np.dot(le, vir[:, None])
                        vr = np.dot(vir[None, :], matr2)
                        vt += 2 * np.dot(vr, lev)

                    D[jj1, jj2] += vt
                    if jj1 != jj2:
                        D[jj2, jj1] += vt

                    rt = np.sum(vsl2 * re.T) / 2  # trace dot
                    if not sym2:
                        rt += np.sum(vsr2 * le.T) / 2  # trace dot

                    hess_re[jj1, jj2] += rt
                    if jj1 != jj2:
                        hess_re[jj2, jj1] += rt

                    if self.reml:
                        ev = np.dot(E, viexog)
                        u1 = np.dot(viexog.T, matl2)
                        u2 = np.dot(matr2.T, ev)
                        um = np.dot(u1, u2)
                        F[jj1][jj2] += um + um.T
                        if not sym2:
                            u1 = np.dot(viexog.T, matr2)
                            u2 = np.dot(matl2.T, ev)
                            um = np.dot(u1, u2)
                            F[jj1][jj2] += um + um.T

        hess_fe -= fac * xtvix / rvir
        hess_re = hess_re - 0.5 * fac * (D / rvir - np.outer(B, B) / rvir ** 2)
        hess_fere = -fac * hess_fere / rvir

        if self.reml:
            QL = [np.linalg.solve(xtvix, x) for x in xtax]
            for j1 in range(self.k_re2 + self.k_vc):
                for j2 in range(j1 + 1):
                    a = np.sum(QL[j1].T * QL[j2])  # trace dot
                    a -= np.trace(np.linalg.solve(xtvix, F[j1][j2]))
                    a *= 0.5
                    hess_re[j1, j2] += a
                    if j1 > j2:
                        hess_re[j2, j1] += a

        # Put the blocks together to get the Hessian.
        m = self.k_fe + self.k_re2 + self.k_vc
        hess = np.zeros((m, m))
        hess[0:self.k_fe, 0:self.k_fe] = hess_fe
        hess[0:self.k_fe, self.k_fe:] = hess_fere.T
        hess[self.k_fe:, 0:self.k_fe] = hess_fere
        hess[self.k_fe:, self.k_fe:] = hess_re

        return hess

    def get_scale(self, fe_params, cov_re, vcomp):
        """
        Returns the estimated error variance based on given estimates
        of the slopes and random effects covariance matrix.

        Parameters
        ----------
        fe_params : array-like
            The regression slope estimates
        cov_re : 2d array-like
            Estimate of the random effects covariance matrix
        vcomp : array-like
            Estimate of the variance components

        Returns
        -------
        scale : float
            The estimated error variance.
        """

        return self._scale

    def set_scale(self, scale):
        """
        Sets the overdispersed scale parameter for the
        exponential family.
        """

        self._scale = scale

    def fit(self, start_params=None, reml=True, niter_sa=0,
            do_cg=True, fe_pen=None, cov_pen=None, free=None,
            full_output=False, method='bfgs', **kwargs):
        """
        """

        _allowed_kwargs = ['gtol', 'maxiter']
        for x in kwargs.keys():
            if x not in _allowed_kwargs:
                raise ValueError("Argument %s not "
                                 "allowed for MixedGLM.fit" % x)

        if method.lower() in ["newton", "ncg"]:
            raise ValueError("method %s not available for MixedGLM" % method)

        self.reml = reml
        self.cov_pen = cov_pen
        self.fe_pen = fe_pen

        self._freepat = free

        if full_output:
            hist = []
        else:
            hist = None

        success = False

        if start_params is None:
            params = MixedGLMParams(self.k_fe, self.k_re, self.k_vc)
            params.fe_params = np.zeros(self.k_fe)
            params.cov_re = np.eye(self.k_re)
            params.vcomp = np.ones(self.k_vc)
        else:
            if isinstance(start_params, MixedGLMParams):
                params = start_params
            else:
                params = MixedGLMParams.from_packed(start_params, self.k_fe,
                                                    self.k_re, self.use_sqrt)

        if do_cg:
            kwargs["retall"] = hist is not None
            if "disp" not in kwargs:
                kwargs["disp"] = False
            packed = params.get_packed(use_sqrt=self.use_sqrt)
            rslt = super(MixedGLM, self).fit(start_params=packed,
                                             skip_hessian=True,
                                             method=method,
                                             **kwargs)

            # The optimization succeeded
            params = np.atleast_1d(rslt.params)
            if hist is not None:
                hist.append(rslt.mle_retvals)

        converged = rslt.mle_retvals['converged']
        if not converged:
            msg = "Gradient optimization failed."
            warnings.warn(msg, ConvergenceWarning)

        # Convert to the final parameterization (i.e. undo the square
        # root transform of the covariance matrix, and the profiling
        # over the error variance).
        params = MixedGLMParams.from_packed(params, self.k_fe, self.k_re,
                                            use_sqrt=self.use_sqrt)
        cov_re_unscaled = params.cov_re
        vcomp_unscaled = params.vcomp
        fe_params = self.get_fe_params(cov_re_unscaled, vcomp_unscaled)
        params.fe_params = fe_params
        scale = self.get_scale(fe_params, cov_re_unscaled, vcomp_unscaled)
        cov_re = scale * cov_re_unscaled
        vcomp = scale * vcomp_unscaled

        if (((self.k_re > 0) and (np.min(np.abs(np.diag(cov_re))) < 0.01)) or
                ((self.k_vc > 0) and (np.min(np.abs(vcomp)) < 0.01))):
            msg = "The MLE may be on the boundary of the parameter space."
            warnings.warn(msg, ConvergenceWarning)

        # Compute the Hessian at the MLE.  Note that this is the
        # Hessian with respect to the random effects covariance matrix
        # (not its square root).  It is used for obtaining standard
        # errors, not for optimization.
        hess = self.hessian(params)
        hess_diag = np.diag(hess)
        if free is not None:
            pcov = np.zeros_like(hess)
            pat = self._freepat.get_packed(use_sqrt=False)
            ii = np.flatnonzero(pat)
            hess_diag = hess_diag[ii]
            if len(ii) > 0:
                hess1 = hess[np.ix_(ii, ii)]
                pcov[np.ix_(ii, ii)] = np.linalg.inv(-hess1)
        else:
            pcov = np.linalg.inv(-hess)
        if np.any(hess_diag >= 0):
            msg = "The Hessian matrix at the estimated " \
                  "parameter values is not positive definite."
            warnings.warn(msg, ConvergenceWarning)

        # Prepare a results class instance
        params_packed = params.get_packed(use_sqrt=False)
        results = MixedGLMResults(self, params_packed, pcov / scale)
        results.params_object = params
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.vcomp = vcomp
        results.scale = scale
        results.cov_re_unscaled = cov_re_unscaled
        results.method = "REML" if self.reml else "ML"
        results.converged = converged
        results.hist = hist
        results.reml = self.reml
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc
        results.use_sqrt = self.use_sqrt
        results.freepat = self._freepat

        return MixedGLMResultsWrapper(results)


class MixedGLMResults(base.LikelihoodModelResults, base.ResultMixin):
    '''
    Class to contain results of fitting a linear mixed effects model.

    MixedGLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Returns
    -------
    **Attributes**

    model : class instance
        Pointer to PHreg model instance that called fit.
    normalized_cov_params : array
        The sampling covariance matrix of the estimates
    fe_params : array
        The fitted fixed-effects coefficients
    re_params : array
        The fitted random-effects covariance matrix
    bse_fe : array
        The standard errors of the fitted fixed effects coefficients
    bse_re : array
        The standard errors of the fitted random effects covariance
        matrix

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''

    def __init__(self, model, params, cov_params):

        super(MixedGLMResults, self).__init__(model, params,
                                              normalized_cov_params=cov_params)
        self.nobs = self.model.nobs
        self.df_resid = self.nobs - np_matrix_rank(self.model.exog)

    @cache_readonly
    def bse_fe(self):
        """
        Returns the standard errors of the fixed effect regression
        coefficients.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(np.diag(self.cov_params())[0:p])

    @cache_readonly
    def bse_re(self):
        """
        Returns the standard errors of the variance parameters.  Note
        that the sampling distribution of variance parameters is
        strongly skewed unless the sample size is large, so these
        standard errors may not give meaningful confidence intervals
        of p-values if used in the usual way.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(self.scale * np.diag(self.cov_params())[p:])

    @cache_readonly
    def random_effects(self):
        """
        Returns the conditional means of all random effects given the
        data.

        Returns
        -------
        random_effects : DataFrame
            A DataFrame with the distinct `group` values as the index
            and the conditional means of the random effects
            in the columns.
        """
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            raise ValueError("Cannot predict random effects from "
                             "singular covariance structure.")

        ranef_dict = {}
        for k in range(self.model.n_groups):

            endog = self.model.endog_li[k]
            exog = self.model.exog_li[k]
            ex_r = self.model.exog_re_li[k]
            ex2_r = self.model.exog_re2_li[k]
            label = self.model.group_labels[k]

            # Get the residuals
            resid = endog
            if self.k_fe > 0:
                expval = np.dot(exog, self.fe_params)
                resid = resid - expval

            solver = _smw_solver(self.scale, ex_r, ex2_r,
                                 self.cov_re, cov_re_inv)
            vresid = solver(resid)

            ranef_dict[label] = np.dot(self.cov_re, np.dot(ex_r.T, vresid))

        column_names = dict(zip(range(self.k_re),
                                self.model.data.exog_re_names))
        df = DataFrame.from_dict(ranef_dict, orient='index')
        return df.rename(columns=column_names).ix[self.model.group_labels]

    @cache_readonly
    def random_effects_cov(self):
        """
        Returns the conditional covariance matrix of the random
        effects for each group given the data.

        Returns
        -------
        random_effects_cov : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional covariance matrix of the
            random effects given the data.
        """

        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        ranef_dict = {}
        # columns = self.model.data.exog_re_names
        for k in range(self.model.n_groups):
            ex_r = self.model.exog_re_li[k]
            ex2_r = self.model.exog_re2_li[k]
            label = self.model.group_labels[k]

            solver = _smw_solver(self.scale, ex_r, ex2_r,
                                 self.cov_re, cov_re_inv)

            mat1 = np.dot(ex_r, self.cov_re)
            mat2 = solver(mat1)
            mat2 = np.dot(mat1.T, mat2)

            ranef_dict[label] = self.cov_re - mat2
            # ranef_dict[label] = DataFrame(self.cov_re - mat2,
            #                              index=columns, columns=columns)

        return ranef_dict

    # Need to override -- t-tests are only used for fixed effects parameters.
    def t_test(self, r_matrix, scale=None, use_t=None):
        """
        Compute a t-test for a each linear hypothesis of the form Rb = q

        Parameters
        ----------
        r_matrix : array-like
            If an array is given, a p x k 2d array or length k 1d
            array specifying the linear restrictions. It is assumed
            that the linear combination is equal to zero.
        scale : float, optional
            An optional `scale` to use.  Default is the scale specified
            by the model fit.
        use_t : bool, optional
            If use_t is None, then the default of the model is used.
            If use_t is True, then the p-values are based on the t
            distribution.
            If use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        res : ContrastResults instance
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.
        """

        if r_matrix.shape[1] != self.k_fe:
            raise ValueError("r_matrix for t-test should have %d columns"
                             % self.k_fe)

        d = self.k_re2 + self.k_vc
        z0 = np.zeros((r_matrix.shape[0], d))
        r_matrix = np.concatenate((r_matrix, z0), axis=1)
        tst_rslt = super(MixedGLMResults, self).t_test(r_matrix,
                                                       scale=scale,
                                                       use_t=use_t)
        return tst_rslt

    def summary(self, yname=None, xname_fe=None, xname_re=None,
                title=None, alpha=.05):
        """
        Summarize the mixed model regression results.

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname_fe : list of strings, optional
            Fixed effects covariate names
        xname_re : list of strings, optional
            Random effects covariate names
        title : string, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """

        from statsmodels.iolib import summary2

        smry = summary2.Summary()

        info = OrderedDict()
        info["Model:"] = "MixedGLM"
        if yname is None:
            yname = self.model.endog_names
        info["No. Observations:"] = str(self.model.n_totobs)
        info["No. Groups:"] = str(self.model.n_groups)

        gs = np.array([len(x) for x in self.model.endog_li])
        info["Min. group size:"] = "%.0f" % min(gs)
        info["Max. group size:"] = "%.0f" % max(gs)
        info["Mean group size:"] = "%.1f" % np.mean(gs)

        info["Dependent Variable:"] = yname
        info["Method:"] = self.method
        info["Scale:"] = self.scale
        info["Likelihood:"] = self.llf
        info["Converged:"] = "Yes" if self.converged else "No"
        smry.add_dict(info)
        smry.add_title("Mixed Linear Model Regression Results")

        float_fmt = "%.3f"

        sdf = np.nan * np.ones((self.k_fe + self.k_re2 + self.k_vc, 6))

        # Coefficient estimates
        sdf[0:self.k_fe, 0] = self.fe_params

        # Standard errors
        sdf[0:self.k_fe, 1] = np.sqrt(np.diag(self.cov_params()[0:self.k_fe]))

        # Z-scores
        sdf[0:self.k_fe, 2] = sdf[0:self.k_fe, 0] / sdf[0:self.k_fe, 1]

        # p-values
        sdf[0:self.k_fe, 3] = 2 * norm.cdf(-np.abs(sdf[0:self.k_fe, 2]))

        # Confidence intervals
        qm = -norm.ppf(alpha / 2)
        sdf[0:self.k_fe, 4] = sdf[0:self.k_fe, 0] - qm * sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 5] = sdf[0:self.k_fe, 0] + qm * sdf[0:self.k_fe, 1]

        # All random effects variances and covariances
        jj = self.k_fe
        for i in range(self.k_re):
            for j in range(i + 1):
                sdf[jj, 0] = self.cov_re[i, j]
                sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
                jj += 1

        # Variance components
        for i in range(self.k_vc):
            sdf[jj, 0] = self.vcomp[i]
            sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
            jj += 1

        sdf = pd.DataFrame(index=self.model.data.param_names, data=sdf)
        sdf.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                       '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
        for col in sdf.columns:
            sdf[col] = [float_fmt % x if np.isfinite(x) else ""
                        for x in sdf[col]]

        smry.add_df(sdf, align='r')

        return smry

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params_object, profile_fe=False)

    @cache_readonly
    def aic(self):
        if self.reml:
            return np.nan
        if self.freepat is not None:
            df = self.freepat.get_packed(use_sqrt=False).sum() + 1
        else:
            df = self.params.size + 1
        return -2 * (self.llf - df)

    @cache_readonly
    def bic(self):
        if self.reml:
            return np.nan
        if self.freepat is not None:
            df = self.freepat.get_packed(use_sqrt=False).sum() + 1
        else:
            df = self.params.size + 1
        return -2 * self.llf + np.log(self.nobs) * df

    def profile_re(self, re_ix, vtype, num_low=5, dist_low=1., num_high=5,
                   dist_high=1.):
        """
        Profile-likelihood inference for variance parameters.

        Parameters
        ----------
        re_ix : integer
            If vtype is `re`, this value is the index of the variance
            parameter for which to construct a profile likelihood.  If
            `vtype` is 'vc' then `re_ix` is the name of the variance
            parameter to be profiled.
        vtype : string
            Either 're' or 'vc', depending on whether the profile
            analysis is for a random effect or a variance component.
        num_low : integer
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : integer
            The number of points at which to calculate the likelihood
            abov the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.

        Returns
        -------
        An array with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.

        Notes
        -----
        Only variance parameters can be profiled.
        """

        pmodel = self.model
        k_fe = pmodel.k_fe
        k_re = pmodel.k_re
        k_vc = pmodel.k_vc
        endog, exog, groups = pmodel.endog, pmodel.exog, pmodel.groups

        # Need to permute the columns of the random effects design
        # matrix so that the profiled variable is in the first column.
        if vtype == 're':
            ix = np.arange(k_re)
            ix[0] = re_ix
            ix[re_ix] = 0
            exog_re = pmodel.exog_re.copy()[:, ix]

            # Permute the covariance structure to match the permuted
            # design matrix.
            params = self.params_object.copy()
            cov_re_unscaled = params.cov_re
            cov_re_unscaled = cov_re_unscaled[np.ix_(ix, ix)]
            params.cov_re = cov_re_unscaled
            ru0 = cov_re_unscaled[0, 0]

            # Convert dist_low and dist_high to the profile
            # parameterization
            cov_re = self.scale * cov_re_unscaled
            low = (cov_re[0, 0] - dist_low) / self.scale
            high = (cov_re[0, 0] + dist_high) / self.scale

        elif vtype == 'vc':
            re_ix = self.model._vc_names.index(re_ix)
            params = self.params_object.copy()
            vcomp = self.vcomp
            low = (vcomp[re_ix] - dist_low) / self.scale
            high = (vcomp[re_ix] + dist_high) / self.scale
            ru0 = vcomp[re_ix] / self.scale

        # Define the sequence of values to which the parameter of
        # interest will be constrained.
        if low <= 0:
            raise ValueError("dist_low is too large and would result in a "
                             "negative variance. Try a smaller value.")
        left = np.linspace(low, ru0, num_low + 1)
        right = np.linspace(ru0, high, num_high + 1)[1:]
        rvalues = np.concatenate((left, right))

        # Indicators of which parameters are free and fixed.
        free = MixedGLMParams(k_fe, k_re, k_vc)
        if self.freepat is None:
            free.fe_params = np.ones(k_fe)
            vcomp = np.ones(k_vc)
            mat = np.ones((k_re, k_re))
        else:
            # If a freepat already has been specified, we add the
            # constraint to it.
            free.fe_params = self.freepat.fe_params
            vcomp = self.freepat.vcomp
            mat = self.freepat.cov_re
            if vtype == 're':
                mat = mat[np.ix_(ix, ix)]
        if vtype == 're':
            mat[0, 0] = 0
        else:
            vcomp[re_ix] = 0
        free.cov_re = mat
        free.vcomp = vcomp

        klass = self.model.__class__
        init_kwargs = pmodel._get_init_kwds()
        if vtype == 're':
            init_kwargs['exog_re'] = exog_re

        likev = []
        for x in rvalues:

            model = klass(endog, exog, **init_kwargs)

            if vtype == 're':
                cov_re = params.cov_re.copy()
                cov_re[0, 0] = x
                params.cov_re = cov_re
            else:
                params.vcomp[re_ix] = x

            # TODO should use fit_kwargs
            rslt = model.fit(start_params=params, free=free,
                             reml=self.reml, cov_pen=self.cov_pen)._results
            likev.append([x * rslt.scale, rslt.llf])

        likev = np.asarray(likev)

        return likev


class MixedGLMResultsWrapper(base.LikelihoodResultsWrapper):
    _attrs = {'bse_re': ('generic_columns', 'exog_re_names_full'),
              'fe_params': ('generic_columns', 'xnames'),
              'bse_fe': ('generic_columns', 'xnames'),
              'cov_re': ('generic_columns_2d', 'exog_re_names'),
              'cov_re_unscaled': ('generic_columns_2d', 'exog_re_names'),
              }
    _upstream_attrs = base.LikelihoodResultsWrapper._wrap_attrs
    _wrap_attrs = base.wrap.union_dicts(_attrs, _upstream_attrs)

    _methods = {}
    _upstream_methods = base.LikelihoodResultsWrapper._wrap_methods
    _wrap_methods = base.wrap.union_dicts(_methods, _upstream_methods)
