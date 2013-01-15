from __future__ import division

from numpy.testing import assert_, assert_raises, dec
#the main drawing function
from statsmodels.sandbox.mosaicplot import mosaic
#other functions to be tested for accuracy
from statsmodels.sandbox.mosaicplot import hierarchical_split,_reduce_dict,_key_splitting,_normalize_split,_split_rect

#utilities for the tests
from collections import Counter,OrderedDict
import numpy
from itertools import product

try:
    import matplotlib.pyplot as pylab
    have_matplotlib = True
except:
    have_matplotlib = False


@dec.skipif(not have_matplotlib)
def plot_mosaic():
    import statsmodels.api as sm
    affairs = sm.datasets.fair.load_pandas()
    datas = affairs.exog
    datas['duration'] = affairs.endog
    categorical = datas[['rate_marriage', 'religious', 'duration']]
    categorical['cheater'] = categorical['duration']>0
    del categorical['duration']
    categorical['rate_marriage'] = categorical['rate_marriage'].map({1: '1 awful',
                        2: '2 bad', 3: '3 intermediate', 4: '4 good', 5: '5 wonderful'})
    categorical['religious'] = categorical['religious'].map({1: 'r1', 2: 'r2', 3: 'r3', 4: 'r4'})
    del categorical['religious']
    data = Counter( tuple(str(k) for k in v.values) for k,v in categorical.iterrows() )
    data = OrderedDict( [k,data[k]] for k in sorted(data.keys()) )
    mosaic(data)
    pylab.title('extraconiugal affairs as function of the marriage status')
    pylab.show()

@dec.skipif(not have_matplotlib)
def plot_mosaic_simple():
    key_set = (['male', 'female'], ['old', 'adult', 'young'],
               ['worker', 'unemployed'], ['healty', 'ill'])
    keys = list(product())
    #data = OrderedDict(zip(keys,[1]*len(keys)))
    data = OrderedDict(zip(keys,range(1,1+len(keys))))
    mosaic(data)
    pylab.title('syntetic data, 4 categories')
    pylab.show()


def plot_mosaic_complex():
    import pylab
    import pandas
    yogurt_url = 'http://vincentarelbundock.github.com/Rdatasets/csv/Ecdat/Yogurt.csv'
    data = pandas.read_csv(yogurt_url,index_col=0)
    data.columns = [ name.replace('.','_') for name in data.columns ]
    names_interesse = ['price_yoplait', 'price_dannon', 'price_hiland', 'price_weight']
    names_ridotti = ['yoplait', 'dannon', 'hiland', 'weight']
    data['cheapest'] = pandas.Series({idx:names_ridotti[numpy.argmin(row)]
                                for idx, row in data[names_interesse].iterrows()})
    count_id = data.groupby(['cheapest','choice'])['id'].count()
    data = dict(count_id)
    mosaic(data,horizontal=False)
    pylab.title('yogurt preferences data')
    pylab.show()


def plot_mosaic_very_complex():
    import pylab
    key_name = ['gender','age','health','work']
    key_base = ['male', 'female'], ['old', 'young'], ['healty', 'ill'], ['work', 'unemployed']
    keys = list(product(*key_base))
    data = OrderedDict(zip(keys, range(1, 1 + len(keys))))
    props = {}
    props[('male', 'old')] = {'color': 'r'}
    props[('female',)] = {'color': 'pink'}
    L = len(key_base)
    fig, axes = pylab.subplots(L, L)
    for i in range(L):
        for j in range(L):
            m = set(range(L)).difference(set((i,j)))
            if i == j:
                axes[i, i].text(0.5, 0.5, key_name[i], ha='center', va='center')
                axes[i, i].set_xticks([])
                axes[i, i].set_xticklabels([])
                axes[i, i].set_yticks([])
                axes[i, i].set_yticklabels([])
            else:
                ji = max(i,j)
                ij = min(i,j)
                temp_data = OrderedDict([((k[ij], k[ji]) + tuple(k[r] for r in m), v)
                                            for k, v in data.items()])
                keys = temp_data.keys()
                for k in keys:
                    value = _reduce_dict(temp_data, k[:2])
                    temp_data[k[:2]] = value
                    del temp_data[k]
                mosaic(temp_data, ax=axes[i, j], colors=props, gap=0.05,horizontal = i > j)
    pylab.show()



eq = lambda x,y:assert_(numpy.allclose(x,y))

def test_recursive_split():
    keys = list(product('mf'))
    data = OrderedDict(zip(keys,[1]*len(keys)))
    res = hierarchical_split(data,gap=0)
    assert_(res.keys()==keys)
    res[('m',)]=(0.0,0.0,0.5,1.0)
    res[('f',)]=(0.5,0.0,0.5,1.0)

    keys = list(product('mf','yao'))
    data = OrderedDict(zip(keys,[1]*len(keys)))
    res = hierarchical_split(data,gap=0)
    assert_(res.keys()==keys)
    res[('m','y')]=(0.0,0.0,0.5,1/3)
    res[('m','a')]=(0.0,1/3,0.5,1/3)
    res[('m','o')]=(0.0,2/3,0.5,1/3)
    res[('f','y')]=(0.5,0.0,0.5,1/3)
    res[('f','a')]=(0.5,1/3,0.5,1/3)
    res[('f','o')]=(0.5,2/3,0.5,1/3)

def test__reduce_dict():
    data = OrderedDict(zip(list(product('mf','oy','wn')),[1]*8))
    eq(_reduce_dict(data,('m',)), 4  )
    eq(_reduce_dict(data,('m','o')), 2  )
    eq(_reduce_dict(data,('m','o','w')), 1  )
    data = OrderedDict(zip(list(product('mf','oy','wn')),range(8)))
    eq(_reduce_dict(data,('m',)), 6  )
    eq(_reduce_dict(data,('m','o')), 1  )
    eq(_reduce_dict(data,('m','o','w')), 0  )

def test__key_splitting():
    #subdivide starting with an empty tuple
    base_rect = { tuple():(0,0,1,1) }
    res = _key_splitting(base_rect,['a','b'],[1,1],tuple(),True,0)
    assert_(res.keys() == [('a',),('b',)])
    eq(res[('a',)],(0,0,0.5,1))
    eq(res[('b',)],(0.5,0,0.5,1))
    #subdivide a in two sublevel
    res_bis = _key_splitting(res,['c','d'],[1,1],('a',),False,0)
    assert_(res_bis.keys() == [('a','c'),('a','d'),('b',)])
    eq(res_bis[('a','c')],(0.0,0.0,0.5,0.5))
    eq(res_bis[('a','d')],(0.0,0.5,0.5,0.5))
    eq(res_bis[('b',)],(0.5,0,0.5,1))
    #starting with a non empty tuple and uneven distribution
    base_rect = { ('total',):(0,0,1,1) }
    res = _key_splitting(base_rect,['a','b'],[1,2],('total',),True,0)
    assert_(res.keys() == [ ('total',)+(e,) for e in  ['a','b']])
    eq(res[('total','a')],(0,0,1/3,1))
    eq(res[('total','b')],(1/3,0,2/3,1))

def test_proportion_normalization():
    # extremes should give the whole set, as well
    # as if 0 is inserted
    eq(_normalize_split(0.), [0.0, 0.0, 1.0])
    eq(_normalize_split(1.), [0.0, 1.0, 1.0])
    eq(_normalize_split(2.), [0.0, 1.0, 1.0])
    #negative values should raise ValueError
    assert_raises(ValueError,_normalize_split,-1)
    assert_raises(ValueError,_normalize_split,[1.,-1])
    assert_raises(ValueError,_normalize_split,[1.,-1,0.])
    #if everything is zero it will complain
    assert_raises(ValueError,_normalize_split,[0.])
    assert_raises(ValueError,_normalize_split,[0.,0.])
    # one-element array should return the whole interval
    eq(_normalize_split([0.5]), [0.0, 1.0])
    eq(_normalize_split([1.]), [0.0, 1.0])
    eq(_normalize_split([2.]), [0.0, 1.0])
    #simple division should give two pieces
    for x in [0.3, 0.5, 0.9]:
        eq(_normalize_split(x), [0.,x,1.0])
    #multiple division should split as the sum of the components
    for x,y in [ (0.25,0.5), (0.1,0.8), (10., 30.) ]:
        eq(_normalize_split([x,y]), [0., x/(x+y), 1.0])
    for x,y,z in [ (1.,1.,1.), (0.1,0.5,0.7), (10., 30., 40) ]:
        eq(_normalize_split([x,y,z]), [0., x/(x+y+z), (x+y)/(x+y+z), 1.0])


def test_false_split():
    #if you ask it to be divided in only one piece, just return the original one
    pure_square = [0., 0., 1., 1. ]
    conf_h = dict(proportion=[1], gap=0.0, horizontal=True)
    conf_v = dict(proportion=[1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square,**conf_h ), pure_square)
    eq(_split_rect(*pure_square,**conf_v ), pure_square)
    conf_h = dict(proportion=[1], gap=0.5, horizontal=True)
    conf_v = dict(proportion=[1], gap=0.5, horizontal=False)
    eq(_split_rect(*pure_square,**conf_h ), pure_square)
    eq(_split_rect(*pure_square,**conf_v ), pure_square)

    #identity on a void rectangle should not give anything strange
    null_square = [0., 0., 0., 0. ]
    conf = dict(proportion=[1], gap=0.0, horizontal=True)
    eq(_split_rect(*null_square,**conf ), null_square)
    conf = dict(proportion=[1], gap=1.0, horizontal=True)
    eq(_split_rect(*null_square,**conf ), null_square)

    #splitting a negative rectangle should raise error
    neg_square = [0., 0., -1., 0. ]
    conf = dict(proportion=[1], gap=0.0, horizontal=True)
    assert_raises(ValueError,_split_rect,*neg_square,**conf)
    conf = dict(proportion=[1,1], gap=0.0, horizontal=True)
    assert_raises(ValueError,_split_rect,*neg_square,**conf)
    conf = dict(proportion=[1], gap=0.5, horizontal=True)
    assert_raises(ValueError,_split_rect,*neg_square,**conf)
    conf = dict(proportion=[1,1], gap=0.5, horizontal=True)
    assert_raises(ValueError,_split_rect,*neg_square,**conf)

def test_rect_pure_split():
    pure_square = [0., 0., 1., 1. ]
    #division in two equal pieces from the perfect square
    h_2split = [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)]
    conf_h = dict(proportion=[1,1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square,**conf_h ), h_2split)

    v_2split = [(0.0, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 0.5)]
    conf_v = dict(proportion=[1,1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square,**conf_v ), v_2split)

    #division in two non-equal pieces from the perfect square
    h_2split = [(0.0, 0.0, 1/3, 1.0), (1/3, 0.0, 2/3, 1.0)]
    conf_h = dict(proportion=[1,2], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square,**conf_h ), h_2split)

    v_2split = [(0.0, 0.0, 1.0, 1/3), (0.0, 1/3, 1.0, 2/3)]
    conf_v = dict(proportion=[1,2], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square,**conf_v ), v_2split)

    #division in three equal pieces from the perfect square
    h_2split = [(0.0, 0.0, 1/3, 1.0), (1/3, 0.0, 1/3, 1.0), (2/3, 0.0, 1/3, 1.0)]
    conf_h = dict(proportion=[1,1,1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square,**conf_h ), h_2split)

    v_2split = [(0.0, 0.0, 1.0, 1/3), (0.0, 1/3, 1.0, 1/3), (0.0, 2/3, 1.0, 1/3)]
    conf_v = dict(proportion=[1,1,1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square,**conf_v ), v_2split)

    #division in three non-equal pieces from the perfect square
    h_2split = [(0.0, 0.0, 1/4, 1.0), (1/4, 0.0, 1/2, 1.0), (3/4, 0.0, 1/4, 1.0)]
    conf_h = dict(proportion=[1,2,1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square,**conf_h ), h_2split)

    v_2split = [(0.0, 0.0, 1.0, 1/4), (0.0, 1/4, 1.0, 1/2), (0.0, 3/4, 1.0, 1/4)]
    conf_v = dict(proportion=[1,2,1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square,**conf_v ), v_2split)

    #splitting on a void rectangle should give multiple void
    null_square = [0., 0., 0., 0. ]
    conf = dict(proportion=[1,1], gap=0.0, horizontal=True)
    eq(_split_rect(*null_square,**conf ), [null_square,null_square])
    conf = dict(proportion=[1,2], gap=1.0, horizontal=True)
    eq(_split_rect(*null_square,**conf ), [null_square,null_square])


def test_rect_deformed_split():
    non_pure_square = [1., -1., 1., 0.5 ]
    #division in two equal pieces from the perfect square
    h_2split = [(1.0, -1.0, 0.5, 0.5), (1.5, -1.0, 0.5, 0.5)]
    conf_h = dict(proportion=[1,1], gap=0.0, horizontal=True)
    eq(_split_rect(*non_pure_square,**conf_h ), h_2split)

    v_2split = [(1.0, -1.0, 1.0, 0.25), (1.0, -0.75, 1.0, 0.25)]
    conf_v = dict(proportion=[1,1], gap=0.0, horizontal=False)
    eq(_split_rect(*non_pure_square,**conf_v ), v_2split)

    #division in two non-equal pieces from the perfect square
    h_2split = [(1.0, -1.0, 1/3, 0.5), (1+1/3, -1.0, 2/3, 0.5)]
    conf_h = dict(proportion=[1,2], gap=0.0, horizontal=True)
    eq(_split_rect(*non_pure_square,**conf_h ), h_2split)

    v_2split = [(1.0, -1.0, 1.0, 1/6), (1.0, 1/6-1, 1.0, 2/6)]
    conf_v = dict(proportion=[1,2], gap=0.0, horizontal=False)
    eq(_split_rect(*non_pure_square,**conf_v ), v_2split)


def test_gap_split():
    pure_square = [0., 0., 1., 1. ]

    #null split
    conf_h = dict(proportion=[1], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square,**conf_h ), pure_square)

    #equal split
    h_2split = [(0.0, 0.0, 0.25, 1.0), (0.75, 0.0, 0.25, 1.0)]
    conf_h = dict(proportion=[1,1], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square,**conf_h ), h_2split)

    #disequal split
    h_2split = [(0.0, 0.0, 1/6, 1.0), (0.5+1/6, 0.0, 1/3, 1.0)]
    conf_h = dict(proportion=[1,2], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square,**conf_h ), h_2split)


if __name__=='__main__':
    #test_proportion_normalization()
    #test_false_split()
    #test_rect_pure_split()
    #test_rect_deformed_split()
    #test_gap_split()
    #test__key_splitting()
    #test__reduce_dict()
    #test_recursive_split()
    plot_mosaic_simple()
    plot_mosaic()
    plot_mosaic_complex()
    plot_mosaic_very_complex()