# Additional installation requirements for travis docbuilds

# Install system dependencies
echo sudo apt-get install graphviz -qq
sudo apt-get install graphviz -qq
# Install required packages and R
echo conda install --channel conda-forge/label/gcc7 sphinx jupyter nbconvert numpydoc "tornado<6" r-robustbase r-lme4 r-geepack libiconv rpy2 --yes --quiet
conda install --channel conda-forge/label/gcc7 sphinx jupyter nbconvert numpydoc "tornado<6" r-robustbase r-lme4 r-geepack libiconv rpy2 --yes --quiet
# doctr and pdr
echo pip install doctr pandas-datareader simplegeneric
pip install doctr pandas-datareader simplegeneric
