from itertools import chain, compress
import numpy as np
from math import ceil, floor
import numbers
import random
import pandas as pd


def check_seed(input_seed):
    if input_seed is None or input_seed is np.random:
        return np.random.mtrand._rand
    
    if isinstance(input_seed, numbers.Integral):
        return np.random.RandomState(input_seed)
    
    if isinstance(input_seed, np.random.RandomState):
        return input_seed
    
    raise ValueError('Incorrect seed')
    
    
def index_data(data, token, token_dtype, axis):
    if hasattr(data, "iloc"):
        if hasattr(token, 'shape'):
            token = np.asarray(token)
            token = token if token.flags.writeable else token.copy()
        elif isinstance(token, tuple):
            token = list(token)
        indexer = data.iloc if token_dtype == 'int' else data.loc
        return indexer[:, token] if axis else indexer[token]
 
    elif hasattr(data, "shape"):
        if token_dtype == 'bool':
            token = np.asarray(token)
        if isinstance(token, tuple):
            token = list(token)
        return data[token] if axis == 0 else data[:, token]
        
    else:
        if np.isscalar(token) or isinstance(token, slice):
            return data[token]
        if token_dtype == 'bool':
            return list(compress(data, token))
        return [data[idx] for idx in token]


def token_type(token, slicing=True):
    err_msg = ("Only Int, Str, Bool allowed")
    dtype_to_str = {int: 'int', str: 'str', bool: 'bool', np.bool_: 'bool'}
    array_dtype_to_str = {'i': 'int', 'u': 'int', 'b': 'bool', 'O': 'str', 'U': 'str', 'S': 'str'}

    if token is None:
        return None
    
    if isinstance(token, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(token)]
        except KeyError:
            raise ValueError(err_msg)

    if isinstance(token, slice):
        if not slicing:
            raise TypeError('Python slice not supported')
            
        if token.start is None and token.stop is None:
            return None
        
        key_start_type = token_type(token.start)
        key_stop_type = token_type(token.stop)
        
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        
        if key_start_type is not None:
            return key_start_type
        
        return key_stop_type
    
    if isinstance(token, (list, tuple)):
        unique_key = set(token)
        key_type = {token_type(elt) for elt in unique_key}
        
        if not key_type:
            return None
        
        if len(key_type) != 1:
            raise ValueError(err_msg)
        
        return key_type.pop()
    
    if hasattr(token, 'dtype'):
        try:
            return array_dtype_to_str[token.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    
    raise ValueError(err_msg)


def indice_indexing(data, indices, *, axis=0):
    if indices is None:
        return data

    if axis not in (0, 1):
        raise ValueError('Axis must be 0 or 1')

    indices_dtype = token_type(indices)

    if axis == 0 and indices_dtype == 'str':
        raise ValueError('String indexing is not supported with axis=0')

    if axis == 1 and data.ndim != 2:
        raise ValueError('Input should be a 2D np array, 2D pandas dataframe')

    if axis == 1 and indices_dtype == 'str' and not hasattr(data, 'loc'):
        raise ValueError('Specifying the columns using strings is only supported for DataFrames' )

    return index_data(data, indices, indices_dtype, axis=axis)
    

def samples(ip):
    if hasattr(ip, 'fit') and callable(ip.fit):
        raise TypeError('Sequence or Array expected')
        
    if not hasattr(ip, '__len__') and not hasattr(ip, 'shape'):
        if hasattr(ip, '__array__'):
            ip = np.asarray(ip)
        else:
            raise TypeError('Sequence or Array expected')

    if hasattr(ip, 'shape') and ip.shape is not None:
        if len(ip.shape) == 0:
            raise TypeError('invalid input')
        if isinstance(ip.shape[0], numbers.Integral):
            return ip.shape[0]
    
    try:
        return len(ip)
    except TypeError as type_error:
        raise TypeError('Sequence or Array expected') from type_error


def data_length(*data):
    sample_len = [samples(i) for i in data if i is not None]
    uniq = np.unique(sample_len)
    if len(uniq) > 1:
        raise ValueError('Inconsistent numbers of samples')
        
        
def make_iter(data):
    if hasattr(data, "__getitem__") or hasattr(data, "iloc"):
        return data

    elif data is None:
        return data

    return np.array(data)


def add_index(*data):
    new_data = [make_iter(i) for i in data]
    data_length(*new_data)
    return new_data


def shuffle_data(data):
    result=[]
    for i in data:
        if hasattr(i, "iloc"):
            new=i.sample(frac=1).reset_index(drop=True)
            result.append(new)
        
        elif hasattr(i, "shape"):
            np.random.shuffle(i)
            result.append(i)
        
        else:
            random.shuffle(i)
            result.append(i)
    
    data_length(*result)
    return result


def data_split(n_samples, test_size, train_size, default_test_size=None):
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (test_size_type == 'i' and (test_size >= n_samples or test_size <= 0) or test_size_type == 'f' and (test_size <= 0 or test_size >= 1)):
        raise ValueError('Incorrect train/test/val size')

    if (train_size_type == 'i' and (train_size >= n_samples or train_size <= 0) or train_size_type == 'f' and (train_size <= 0 or train_size >= 1)):
        raise ValueError('Incorrect train/test/val size')

    if train_size is not None and train_size_type not in ('i', 'f'):
        raise ValueError('Incorrect train/test/val size')
        
    if test_size is not None and test_size_type not in ('i', 'f'):
        raise ValueError('Incorrect train/test/val size')

    if (train_size_type == 'f' and test_size_type == 'f' and train_size + test_size > 1):
        raise ValueError('Incorrect train/test/val size')

    if test_size_type == 'f':
        n_test = ceil(test_size * n_samples)
    
    elif test_size_type == 'i':
        n_test = float(test_size)

    if train_size_type == 'f':
        n_train = floor(train_size * n_samples)
    
    elif train_size_type == 'i':
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('Incorrect train/test/val size')

    n_train, n_test = int(n_train), int(n_test)
    
    if n_train == 0:
        raise ValueError('Incorrect train/test/val size')

    return n_train, n_test


def classes_index(y):
    
    y = y_check(y)
    
    unique_list = []
    for i in y:
        if i not in unique_list:
            unique_list.append(i)
    
    if len(unique_list) < 2:
        raise ValueError('Cannot use stratify for 1 class')
    
    counter=[]
    for j in unique_list:
        counter.append(y.count(j))
    
    minval = min(counter)
    
    newlst=[]
    for x in unique_list:
        class_index=[]
        class_index = [i for i,j in enumerate(y) if y[i] == x]
        for i in range(minval):
            newlst.append(class_index[i])
    
    finlist=[]
    for i in range(int(len(newlst)/2)):
        finlist.append(newlst[i])
        finlist.append(newlst[len(newlst)-i-1])
        
    return finlist, y


def y_check(y):
    if hasattr(y, "iloc") or hasattr(y, "shape"):
        newy=y.tolist()
        return newy
    else:
        return y
 
            
def reduce_data(data,ci_list):
    if hasattr(data, "iloc"):
        newdata = data.iloc[ci_list, : ].reset_index()
        newdata = newdata.drop(['index'],axis=1)
        return newdata
 
    elif hasattr(data, "shape"):
        newdata = data[ci_list,:]
        return newdata
        
    else:
        newdata = [data[i] for i in ci_list]
        return newdata


def reduce(*data,class_index_list):
    res = [reduce_data(i,class_index_list) for i in data]
    data_length(*res)
    return res


def y_transform(y,ytemp):
    if hasattr(y, "iloc"):
        return pd.DataFrame(ytemp)
 
    elif hasattr(y, "shape"):
        return np.array(ytemp)
        
    else:
        return ytemp 


def train_test_split(*data, test_size=None, train_size=None, val_size=None, val_split=False, shuffle=True, stratify=False, random_state=None):
    n = len(data)
    if n == 0:
        raise ValueError("Array expected")
    
    data = add_index(*data)
    n_samples = samples(data[0])
    y=data[n-1]
    
    if shuffle == True:
        rnum=check_seed(random_state)
        random_state = rnum.permutation(n_samples)
        data=shuffle_data(data)
    
    if stratify == True:
        y=data[n-1]
        class_index_list , newy = classes_index(y)
        data[n-1] = newy
        data = reduce(*data,class_index_list=class_index_list)
        n_samples = samples(data[0])
        shuffle = False
     
    data[n-1] = y_transform(y,data[n-1])
        
    if val_split == True:
        
        if val_size != None and train_size != None:
            new_size = round(val_size + train_size,4)
            n_trainval, n_test = data_split(n_samples, test_size, new_size, default_test_size=0.2)
            
            newval_size = round(val_size/(train_size + val_size),4)
            newtrain_size = round(train_size/(train_size + val_size),4)
            n_train, n_val = data_split(n_trainval, newval_size, newtrain_size, default_test_size=0.125)
        else:
            n_trainval, n_test = data_split(n_samples, test_size, train_size, default_test_size=0.2)
            n_train, n_val = data_split(n_trainval, val_size, train_size, default_test_size=0.125)
        
        train = np.arange(n_train)
        test = np.arange(n_trainval, n_trainval + n_test)
        validation = np.arange(n_train, n_train + n_val)
        
        return list(chain.from_iterable((indice_indexing(a, train), indice_indexing(a, test), indice_indexing(a, validation)) for a in data))
    
    else:
        n_train, n_test = data_split(n_samples, test_size, train_size, default_test_size=0.25)

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)
   
        return list(chain.from_iterable((indice_indexing(a, train), indice_indexing(a, test)) for a in data))

