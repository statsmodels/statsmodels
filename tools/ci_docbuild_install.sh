#!/usr/bin/env bash

set -x  # echo on
# Install required packages
conda install sphinx ipython jupyter nbconvert numpydoc --yes --quiet
pip install pandas-datareader colorama
conda install -c r rpy2 --yes --quiet
