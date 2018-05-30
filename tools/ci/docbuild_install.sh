# Additional installation requirements for travis docbuilds

set -x  # echo on
# Install system dependencies
sudo apt-get install graphviz -qq
# Install required packages
conda install sphinx ipython jupyter nbconvert numpydoc --yes --quiet
pip install git+https://github.com/pydata/pandas-datareader.git
pip install colorama doctr
conda install -c r rpy2 r-robustbase --yes --quiet
