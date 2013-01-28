import numpy as np
import pandas as pd

def _array_to_ascii(ar, align_data='r', align_index='l', align_header='c', 
        sep_table_above=False, sep_table_below=False, sep_table_char='=',
        sep_header_below=True, sep_header_above=True, sep_header_char='-',
        pad_col='  ', pad_index=None, fixed_width=False):
    '''Convert numpy string array to ascii table'''

    if fixed_width:
        max_len_table = []
        for col in range(ar.shape[1]):
            max_len_table.append(max([len(x.strip()) for x in ar[:,col]]))
        max_len_table = max(max_len_table)

    for col in range(ar.shape[1]):
        if fixed_width: 
            max_len = max_len_table
        else:
            max_len = max([len(x.strip()) for x in ar[:,col]])
        if align_data == 'c':
            ar[:,col] = [x.strip().center(max_len) for x in ar[:,col]]
        elif align_data == 'l':
            ar[:,col] = [x.strip().ljust(max_len) for x in ar[:,col]]
        elif align_data == 'r':
            ar[:,col] = [x.strip().rjust(max_len) for x in ar[:,col]]
        else:
            raise Exception('align_data must be l, c or r')

    if align_header != None:
        if align_header == 'c':
            ar[0,:] = [x.strip().center(len(x)) for x in ar[0,:]]
        elif align_header == 'l':
            ar[0,:] = [x.strip().ljust(len(x)) for x in ar[0,:]]
        elif align_header == 'r':
            ar[0,:] = [x.strip().rjust(len(x)) for x in ar[0,:]]
        else:
            raise Exception('align_header must be l, c or r')

    if align_index != None:
        if align_index == 'c':
            ar[:,0] = [x.strip().center(len(x)) for x in ar[:,0]]
        elif align_index == 'l':
            ar[:,0] = [x.strip().ljust(len(x)) for x in ar[:,0]]
        elif align_index == 'r':
            ar[:,0] = [x.strip().rjust(len(x)) for x in ar[:,0]]
        else:
            raise Exception('align_index must be l, c or r')

    tab = ar.tolist()

    if pad_index != None:
        for t in tab:
            t[0] = t[0] + pad_index

    tab = [pad_col.join(x) for x in tab]
    max_len = max([len(x) for x in tab])
    sep_table = max_len * sep_table_char
    sep_header = max_len * sep_header_char
    if sep_header_above:
        tab[0] = sep_header + '\n' + tab[0]
    if sep_header_below: 
        tab[0] = tab[0] + '\n' + sep_header
    if sep_table_above:
        tab[0] = sep_table + '\n' + tab[0]
    if sep_table_below: 
        tab[-1] = tab[-1] + '\n' + sep_table

    tab = '\n'.join(tab)
    return tab

def _df_to_ascii(df, header=True, index=True, float_format="%.4f", 
        align_data='r', align_index='l', align_header='c', 
        sep_table_above=False, sep_table_below=False, sep_table_char='=',
        sep_header_above=True, sep_header_below=True, sep_header_char='-',
        pad_col='  ', pad_index=None, fixed_width=False):
    '''Convert a Pandas DataFrame to a numpy string array and convert
    to ascii table'''

    dataframe = df.copy()

    def forg(x):
        try:
            return float_format % x
        except:
            return str(x)
    f = lambda x: x.apply(forg)
    dataframe = dataframe.apply(f)
    ar = np.array(dataframe).tolist() 

    if header:
        cols = dataframe.columns.tolist()
        cols = [str(x) for x in cols]
        ar = [cols] + ar
        if index:
            idx = [''] + dataframe.index.tolist()
            idx = [str(x) for x in idx]
            for i,v in enumerate(idx):
                ar[i] = [v] + ar[i]
    else: 
        sep_header_below = False
        if index:
            idx = dataframe.index.tolist()
            idx = [str(x) for x in idx]
            for i,v in enumerate(idx):
                ar[i] = [v] + ar[i]
    ar = np.array(ar)
    tab = _array_to_ascii(ar, align_data=align_data, align_index=align_index,
            align_header=align_header, sep_table_above=sep_table_above,
            sep_table_below=sep_table_below, sep_table_char=sep_table_char,
            sep_header_above=sep_header_above,
            sep_header_below=sep_header_below, sep_header_char=sep_header_char,
            pad_col=pad_col, pad_index=pad_index, fixed_width=fixed_width) 
    return tab
