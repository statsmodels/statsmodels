# Additional installation requirements for travis docbuilds

# Install system dependencies
echo sudo apt-get update
sudo apt-get update

echo sudo apt-get install graphviz libgfortran3 enchant -qq
sudo apt-get install graphviz libgfortran3 enchant -qq
# Install required packages and R
echo conda install --channel conda-forge sphinx jupyter nbconvert numpydoc==0.8.0 r-robustbase r-lme4 r-geepack libiconv rpy2 --yes --quiet
conda install --channel conda-forge sphinx jupyter nbconvert numpydoc==0.8.0 r-robustbase r-lme4 r-geepack libiconv rpy2 --yes --quiet
# doctr and pdr
echo pip install doctr pandas-datareader simplegeneric seaborn sphinxcontrib-spelling
pip install doctr pandas-datareader simplegeneric seaborn sphinxcontrib-spelling
