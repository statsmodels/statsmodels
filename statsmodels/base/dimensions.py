#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

dimensions.py defines mixin classes for pinning down Model/Results attributes
in terms of each other.  In particular, these attributes are those
that depend on the *shape* of the data or the estimation method being
used, but *not* the data themselves.

Examples:
	df_model, df_resid
	nobs
	neqs
	k_exog, k_endog

"""
import numpy as np
import bottleneck as bn

from statsmodels.tools.decorators import cache_readonly



def _get_k_constant(inst, assume_const, assert_const=False):
	"""
	There is a long-existing eyesore in statsmodels of adding 1.0
	to `df_model` along with a comment "# assumes constant"

	_get_k_constant exists to check that assumption, handle it
	systematically, and eventually deprecate it.

	The intention as designed is for `_assume_const` and `_assert_const` to
	be class attributes belong to `inst`.  Then calling would look like

	>>> k_const = _get_k_constant(self, self._assume_const, self._assert_const)

	This pattern is encouraged but not required.


	If `assume_const` argument is True, the value `1.0` is always returned.
	Otherwise, `inst.k_constant` is returned.

	**
	The goal is for as many classes as possible to set `_assume_constant`
	to `False`
	**

	If the `assert_const` argument is True, we assert that
	`inst.k_constant == 1`.  If this assertion is made *and passes* then
	we can infer that this class does not *need* to assume there is a
	constant.  This class can be cleaned up by eliminating this behavior.
	Retaining the `assert_const` in future runs behaves as a regression
	test.


	# See GH #1624, #1664, #1719, #1724, #1930, #3574
	# Note 1624 is a semi-separate issue that is a *different* eyesore
	"""
	if assume_const:
		k_const = 1.0
	else:
		k_const = inst.k_constant

	if assert_const:
		assert inst.k_constant == 1, inst.k_constant
	return k_const


# Inherited by Model and Results
class NEQsMixin(object):
	
	@cache_readonly
	def neqs(self):
		try:
			# Note: Recall that MLEModel and sub-classes have endog transposed
			# TODO: does the above apply to any of DiscreteModel?
			return self.endog.shape[1] # FIXME: does the MLEModel note above mean that this will be wrong?
		except (AttributeError, IndexError):
			# Note: np.ndim(None) is 0
			return np.ndim(self.endog)

	k_endog = neqs # TODO: Is this a misnomer for MultinomialModel, MNLogit?


# Inherited by VARResults, Model, and Results
class NobsMixin(object):
	@cache_readonly
	def _nobs_total(self):
		"""
		AR, ARIMA, VAR, ... models have `nobs` that is not always equal
		to `len(endog)`.  `_nobs_total` gives the *total* observations,
		including any that are dropped or "burned" like in these special
		cases.
		"""
		try:
			# regime_switching
			return len(self.orig_endog) # TODO: should we use self.data.orig_endog?
		except AttributeError:
			return len(self.endog)
			# FIXME: This may break in models where self.endog is messed with.

	@cache_readonly
	def nobs(self):
		# Note: we could use endog.shape[0], but in at least
		# one test case self.endog is a list
		return len(self.endog)
		# Notes from the GEE Case:
		# v1 = sum([len(y) for y in self.endog_li])
		# v2 = sum([len(y) for y in self.group_indices.values()])
		# v3 = len(self.endog)
		# assert v1 == v2, (v1, v2)
		# assert v3 == v1, (v3, v1)


class KarNobsMixin(object):
	@property
	def nobs(self):
		# In markov_switching and markov_autoregression, _nobs_total comes
		# from len(self.orig_endog).  markov_switching does not have a
		# k_ar attribute, but markov_autoregression (which subclasses markov_swiching)
		# does.
		#
		# In linear_model, this is equivalent to self.wexog.shape[0].
		# For OLS, WLS, and GLS this is the same as _nobs_total, i.e.
		# len(self.endog).  However for GLSAR the `whiten` method
		# returns a `self.wexog` which is shorter than `self.endog` by `k_ar`,
		# so we subtract that term to get `nobs`.
		return self._nobs_total - getattr(self, 'k_ar', 0)


class CssKarNobsMixin(object):
	@property
	def nobs(self):
		n_totobs = self._nobs_total
		if self.method in ["cmle", "css", "ols"]:
			nobs = (n_totobs - self.k_ar)
		else:
			nobs = n_totobs
		# ARModel used to set nobs = n_totobs iff method == "mle" and
		# otherwise raise NotImplementedError
		return float(nobs)


class WNobsMixin(object):
	"""

	This mixin is specific to GLM Model, defines a `wnobs` property in
	terms of the model's `freq_weights` attribute.

	"""

	@property
	def wnobs(self):
		if (self.freq_weights is not None) and (self.freq_weights.shape[0] == self.endog.shape[0]):
			wnobs = bn.nansum(self.freq_weights) # self.freq_weights.sum()
		else:
			wnobs = self.exog.shape[0] ## wexog??  # TODO: not hit in tests
		return wnobs


class KTrendMixin(object):
	# Note: making k_exog a cache_readonly or cached_property breaks
	# a couple of ARIMA tests; not sure why.
	@property
	def k_trend(self): # Almost Equiv: tsa.vector_ar.util.get_trendorder(trend)
		trend = self.trend
		if trend in ['n', 'nc', None]:
			kt = 0
		elif trend == 'c': # I think only 0 and 1 are implemented  ## should this be related to k_constant?
			kt = 1
		elif trend == 't':  # TODO: not hit in tests
			kt = 1
		elif trend == 'ct':
			kt = 2
		elif isinstance(trend, (list, np.array)):
			# e.g. SARIMAX
			kt = sum(trend)
		else:
			raise NotImplementedError(trend)
			#kt = 0
		return kt


# FIXME: What if self.exog exists but self.data has not been set yet?
class KExogMixin(object):

	# Note: making k_exog a cache_readonly breaks
	# a couple of ARIMA tests; not sure why.
	@property
	def k_exog(self):
		try:
			# FIXME: Could this be wrong depending on whether the user already passed a trend or constant?
			# TODO: What about C_CONTIGUOUS/F_CONTIGUOUS?
			return self.data.orig_exog.shape[1]
			# For most models:
			# 	equiv: self.exog.shape[1]
			#
			# For scalar AR/ARMA/ARIMA models:
			# 	equiv: return self.exog.shape[1] - self.k_trend
			#
			# In these cases the `exog` attribute includes
			# possible trend columns that are added internally
			# (i.e. not part of the `exog` passed to __init__)
			# To get at the intended `k_exog`, we need to subtract `k_trend`
			# from self.exog.shape[1].  Alternatively, we can just check
			# `self.data.orig_exog.shape[1]`
		except AttributeError:
			# There is no `self.exog` (or more specifically, `self.data.orig_exog`)
			return 0
		except IndexError:
			# IndexError: tuple index out of range
			# If orig_exog is 1-dimensional, we just want 1.  If it is
			# zero-dimensional, we just want 0.  It is a coincidence that
			# np.ndim(...) gives the desired measurement.
			return np.ndim(self.data.orig_exog)
			# FIXME: is this going to incorrectly return 0 when we have a scalar instead of an array?


class MNDimensions(object):
	@cache_readonly
	def J(self):
		# number of alternative choices
		return self.wendog.shape[1]

	# TODO: Is this equivalent to k_exog?
	@cache_readonly
	def K(self):
		# number of variables
		return self.exog.shape[1]


class L1Estimator(object):
	_assume_const = True
	_assert_const = True

	@cache_readonly
	def trimmed(self):
		"""trimmed is a boolean array with T/F telling whether or not that
		entry in params has been set zero'd out."""
		return self.mle_retvals['trimmed']

	@cache_readonly
	def nnz_params(self):
		"""Number of non-zero params"""
		return (self.trimmed == False).sum()


