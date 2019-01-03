# Additional installation requirements for travis docbuilds

# Install system dependencies
echo sudo apt-get install graphviz -qq
sudo apt-get install graphviz -qq
# Install required packages and R
echo conda install --channel conda-forge/label/gcc7 sphinx ipython jupyter nbconvert numpydoc tzlocal r-robustbase r-lme4 r-geepack libiconv --yes --quiet
conda install --channel conda-forge/label/gcc7 sphinx ipython jupyter nbconvert numpydoc tzlocal r-robustbase r-lme4 r-geepack libiconv --yes --quiet
# doctr and pdr
echo pip install colorama doctr pandas-datareader rpy2
pip install colorama doctr pandas-datareader rpy2
