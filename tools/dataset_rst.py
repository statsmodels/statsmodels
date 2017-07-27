#!/usr/bin/env python
"""
Run this script to convert dataset documentation to ReST files. Relies
on the meta-information from the datasets of the currently installed version.
Ie., it imports the datasets package to scrape the meta-information.
"""

import statsmodels.api as sm
import os
from os.path import join, realpath, dirname
import inspect
from string import Template

import hash_funcs

file_path = dirname(__file__)
dest_dir = realpath(join(file_path, '..', 'docs', 'source', 'datasets',
                         'generated'))

datasets = dict(inspect.getmembers(sm.datasets, inspect.ismodule))
datasets.pop('utils')

doc_template = Template(u"""$TITLE
$title_

Description
-----------

$DESCRIPTION

Notes
-----
$NOTES

Source
------
$SOURCE

Copyright
---------

$COPYRIGHT
""")

if __name__ == "__main__":

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for dataset in datasets:
        write_pth = join(dest_dir, dataset + '.rst')
        data_mod = datasets[dataset]
        title = getattr(data_mod, 'TITLE')
        descr = getattr(data_mod, 'DESCRLONG')
        copyr = getattr(data_mod, 'COPYRIGHT')
        notes = getattr(data_mod, 'NOTE')
        source = getattr(data_mod, 'SOURCE')
        write_file = doc_template.substitute(TITLE=title,
                                             title_='='*len(title),
                                             DESCRIPTION=descr, NOTES=notes,
                                             SOURCE=source, COPYRIGHT=copyr)
        to_write, filehash = hash_funcs.check_hash(write_file.encode(),
                                                   data_mod.__name__.encode())
        if not to_write:
            print("Hash has not changed for docstring of dataset "
                  "{}".format(dataset))
            continue
        with open(os.path.realpath(write_pth), 'w') as rst_file:
            rst_file.write(write_file)
        if filehash is not None:
            hash_funcs.update_hash_dict(filehash, data_mod.__name__)
