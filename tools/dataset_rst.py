#!/usr/bin/env python3
"""
Run this script to convert dataset documentation to ReST files. Relies
on the meta-information from the datasets of the currently installed version.
Ie., it imports the datasets package to scrape the meta-information.
"""

import glob
import inspect
import os
from os.path import join, realpath, dirname
from string import Template

import statsmodels.api as sm

file_path = dirname(__file__)
dest_dir = realpath(join(file_path, '..', 'docs', 'source', 'datasets',
                         'generated'))

datasets = dict(inspect.getmembers(sm.datasets, inspect.ismodule))
datasets.pop('utils')
last_mod_time = {}
for dataset in datasets:
    root = os.path.abspath(os.path.split(datasets[dataset].__file__)[0])
    files = glob.glob(os.path.join(root, '*'))
    if not files:
        raise NotImplementedError('Must be files to read the date')
    mtime = 0.0
    for f in files:
        if f.startswith('__') and f != '__init__.py':
            continue
        mtime = max(mtime, os.path.getmtime(f))
    last_mod_time[dataset] = mtime

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

$COPYRIGHT\
""")

if __name__ == "__main__":

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for dataset in datasets:
        rst_file_name = dataset + '.rst'
        write_pth = join(dest_dir, rst_file_name)
        if os.path.exists(write_pth):
            rst_mtime = os.path.getmtime(write_pth)
            if rst_mtime > last_mod_time[dataset]:
                print('Skipping creation of {0} since the rst file is newer '
                      'than the data files.'.format(rst_file_name))
                continue
        data_mod = datasets[dataset]
        title = getattr(data_mod, 'TITLE')
        descr = getattr(data_mod, 'DESCRLONG')
        copyr = getattr(data_mod, 'COPYRIGHT')
        notes = getattr(data_mod, 'NOTE')
        source = getattr(data_mod, 'SOURCE')
        write_file = doc_template.substitute(TITLE=title,
                                             title_='=' * len(title),
                                             DESCRIPTION=descr, NOTES=notes,
                                             SOURCE=source, COPYRIGHT=copyr)
        print('Writing {0}.'.format(rst_file_name))
        with open(os.path.realpath(write_pth), 'w') as rst_file:
            rst_file.write(write_file)
