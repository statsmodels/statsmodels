# -*- coding: utf-8 -*-

from statsmodels.tools import decorators
from statsmodels.tools.decorators import copy_doc


class TestCopyDoc:

	def test_copy_doc(self):
		
		@copy_doc("foo")
		def func(*args, **kwargs):
			return (args, kwargs)

		assert func.__doc__ == "foo"


	def test_copy_doc_overwrite(self):
		
		@copy_doc("foo")
		def func(*args, **kwargs):
			"""bar"""
			return (args, kwargs)

		assert func.__doc__ == "foo"


	def test_copy_doc_orig_altered(self):

		def func(*args, **kwargs):
			"""bar"""
			return (args, kwargs)

		func2 = copy_doc("foo")(func)
		assert func2.__doc__ == "foo"
		assert func.__doc__ == "foo"
