#! /usr/bin/env python
"""
Run this script to convert dataset documentation to ReST files. Relies
on the meta-information from the datasets of the currently installed version.
Ie., it imports the datasets package to scrape the meta-information.
"""

import scikits.statsmodels.api as sm
import os
from os.path import join
import inspect
from string import Template

datasets = dict(inspect.getmembers(sm.datasets, inspect.ismodule))
datasets.pop('datautils')
datasets.pop('nile') #TODO: fix docstring in nile

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

for dataset in datasets:
    write_pth = join('../scikits/statsmodels/docs/source/datasets/generated',
                             dataset+'.rst')
    data_mod = datasets[dataset]
    with open(os.path.realpath(write_pth), 'w') as rst_file:
        title = getattr(data_mod,'TITLE')
        descr = getattr(data_mod, 'DESCRLONG')
        copyr = getattr(data_mod, 'COPYRIGHT')
        notes = getattr(data_mod, 'NOTE')
        source = getattr(data_mod, 'SOURCE')
        write_file = doc_template.substitute(TITLE=title,
                                             title_='='*len(title),
                                             DESCRIPTION=descr, NOTES=notes,
                                             SOURCE=source, COPYRIGHT=copyr)
        rst_file.write(write_file)
