# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

# The most simple use case is to take a dictionary and plot the result
data = {'a': 10, 'b': 15, 'c': 16}
mosaic(data, title='basic dictionary')
plt.show()

# A more useful example is given by a dictionary with multiple indices.  In
# this case we use a wider gap to a better visual separation of the resulting
# plot
data = {('a', 'b'): 1, ('a', 'c'): 2, ('d', 'b'): 3, ('d', 'c'): 4}
mosaic(data, gap=0.05, title='complete dictionary')
plt.show()

# The same data can be given as a simple or hierarchical indexed Series
rand = np.random.random
tuples = list(product(['bar', 'baz', 'foo', 'qux'], ['one', 'two']))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
data = pd.Series(rand(8), index=index)
mosaic(data, title='hierarchical index series')
plt.show()

# The third accepted data structureis the np array, for which a very simple
# index will be created.
rand = np.random.random
data = 1+rand((2, 2))
mosaic(data, title='random non-labeled array')
plt.show()

# If you need to modify the labeling and the coloring you can give a function
# to create the labels and one with the graphical properties starting from the
# key tuple


def props(key):
    return {'color': 'r' if 'a' in key else 'gray'}


def labelizer(key):
    return {('a',): 'first', ('b',): 'second', ('c',): 'third'}[key]


data = {'a': 10, 'b': 15, 'c': 16}
mosaic(data, title='colored dictionary', properties=props, labelizer=labelizer)
plt.show()

# Using a DataFrame as source, specifying the name of the columns of interest
gender = ['male', 'male', 'male', 'female', 'female', 'female']
pet = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']
data = pd.DataFrame({'gender': gender, 'pet': pet})
mosaic(data, ['pet', 'gender'], title='DataFrame as Source')
plt.show()
