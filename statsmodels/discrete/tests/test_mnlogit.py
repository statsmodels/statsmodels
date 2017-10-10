
import os

import numpy as np
import pandas as pd

import pytest

import statsmodels.api as sm



here = os.path.dirname(__file__)


def test_mailing_list():
	# https://groups.google.com/forum/#!topic/pystatsmodels/FAnjBe_RynE
	path = os.path.join(here, 'Sample.xlsx')

	df = pd.read_excel(path, index_col=False)

	endog = df['Count']
	exog = sm.add_constant(df[['X', 'Y', 'Z']])
	model = sm.MNLogit(endog=endog, exog=exog)

	res = model.fit()

	assert np.isnan(res.params.values).all()
	# status quo 2017-10-09, not desirable

	with pytest.raises(ValueError):
		# status quo 2017-10-09, not desirable
		# ValueError: Must pass 2-d input
		res.conf_int()

	with pytest.raises(ValueError):
		# status quo 2017-10-09, not desirable
		# ValueError: operands could not be broadcast together with
		# shapes (4,) (4,7)
		res.summary2()


def test_categorical():
	path = os.path.join(here, 'Sample.xlsx')

	df = pd.read_excel(path, index_col=False)

	names = ['Aragorn', 'Gandalf', 'Legolas', 'Boromir', 'Gimli',
			 'Samwise', 'Frodo', 'Merry', 'Pip']
	endog = df['Count'].apply(lambda n: names[n-1]).astype('category')
	
	exog = sm.add_constant(df[['X', 'Y', 'Z']])
	model = sm.MNLogit(endog=endog, exog=exog)

	res = model.fit()

	assert (res.params.index == exog.columns).all()
	assert (res.params.columns == endog.cat.categories).all()
	# This will fail if there are dropped categories

