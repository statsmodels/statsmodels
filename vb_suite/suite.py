from __future__ import print_function

from vbench.api import Benchmark
from datetime import datetime

import os

modules = ['arima', 'discrete', 'ols']

by_module = {}
benchmarks = []

for modname in modules:
    ref = __import__(modname)
    by_module[modname] = [v for v in ref.__dict__.values()
                          if isinstance(v, Benchmark)]
    benchmarks.extend(by_module[modname])

for bm in benchmarks:
    assert(bm.name is not None)

import getpass
import sys

USERNAME = getpass.getuser()

if sys.platform == 'darwin':
    HOME = '/Users/%s' % USERNAME
else:
    HOME = '/home/%s' % USERNAME

try:
    import ConfigParser

    config = ConfigParser.ConfigParser()
    config.readfp(open(os.path.expanduser('~/.sm_vbenchcfg')))

    REPO_PATH = config.get('setup', 'repo_path')
    REPO_URL = config.get('setup', 'repo_url')
    DB_PATH = config.get('setup', 'db_path')
    TMP_DIR = config.get('setup', 'tmp_dir')
except:
    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    REPO_URL = 'https://github.com/statsmodels/statsmodels.git'
    DB_PATH = os.path.join(REPO_PATH, 'vb_suite/benchmarks.db')
    TMP_DIR = os.path.join(HOME, 'tmp/vb_statsmodels')


PREPARE = """
python setup.py clean
"""
BUILD = """
python setup.py build_ext --inplace
"""

dependencies = ['sm_vb_common.py']

START_DATE = datetime(2013, 6, 1)

RST_BASE = 'source'

# HACK!

# timespan = [datetime(2011, 1, 1), datetime(2012, 1, 1)]


def generate_rst_files(benchmarks):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    vb_path = os.path.join(RST_BASE, 'vbench')
    fig_base_path = os.path.join(vb_path, 'figures')

    if not os.path.exists(vb_path):
        print('creating {}'.format(vb_path))
        os.makedirs(vb_path)

    if not os.path.exists(fig_base_path):
        print('creating {}'.format(fig_base_path))
        os.makedirs(fig_base_path)

    for bmk in benchmarks:
        print('Generating rst file for {}'.format(bmk.name))
        rst_path = os.path.join(RST_BASE, 'vbench/%s.txt' % bmk.name)

        fig_full_path = os.path.join(fig_base_path, '%s.png' % bmk.name)

        # make the figure
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        bmk.plot(DB_PATH, ax=ax)

        start, end = ax.get_xlim()

        plt.xlim([start - 30, end + 30])
        plt.savefig(fig_full_path, bbox_inches='tight')
        plt.close('all')

        fig_rel_path = 'vbench/figures/%s.png' % bmk.name
        rst_text = bmk.to_rst(image_path=fig_rel_path)
        with open(rst_path, 'w') as f:
            f.write(rst_text)

    with open(os.path.join(RST_BASE, 'index.rst'), 'w') as f:
        print("""
Performance Benchmarks
======================

These historical benchmark graphs were produced with `vbench
<http://github.com/pydata/vbench>`__.

The ``statsmodels_vb_common`` setup script can be found here_

.. _here: https://github.com/statsmodels/statsmodels/tree/master/vb_suite

Produced on a machine with:

  * Intel Core 2 Quad Q9550 2.83 Ghz Processor
  * Kubuntu Linux 13.10
  * Python 2.7.5+
  * NumPy 1.8.0.dev-8e0a542 (with ATLAS 3.9.76)
  * SciPy 0.14.0-dev-1b7c11c
  * Pandas 0.13.1-311-g63f46c1
  * Patsy 0.1.0

.. toctree::
   :maxdepth: 3
  """, file=f)
        for modname, mod_bmks in sorted(by_module.items()):
            print('   vb_{}'.format(modname), file=f)
            modpath = os.path.join(RST_BASE, 'vb_%s.rst' % modname)
            with open(modpath, 'w') as mh:
                header = '{0}\n{1}\n\n'.format(modname, '=' * len(modname))
                print(header, file=mh)

                for bmk in mod_bmks:
                    print(bmk.name, file=mh)
                    print('-' * len(bmk.name), file=mh)
                    print('.. include:: vbench/{}.txt\n'.format(bmk.name),
                          file=mh)
