# Additional installation requirements for travis docbuilds

# Install system dependencies
echo sudo apt-get update
sudo apt-get update

echo sudo apt-get install graphviz libgfortran3 -qq
sudo apt-get install graphviz libgfortran3 -qq
# Install required packages and R
echo conda install --channel conda-forge sphinx jupyter nbconvert numpydoc r-robustbase r-lme4 r-geepack libiconv rpy2 --yes --quiet
conda install --channel conda-forge sphinx jupyter nbconvert numpydoc r-robustbase r-lme4 r-geepack libiconv rpy2 --yes --quiet
# doctr and pdr
echo pip install doctr pandas-datareader simplegeneric
pip install doctr pandas-datareader simplegeneric
