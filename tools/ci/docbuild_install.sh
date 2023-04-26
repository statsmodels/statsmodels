# Additional installation requirements for travis docbuilds

# Install system dependencies
echo sudo apt-get update
sudo apt-get update

echo sudo apt-get install graphviz libgfortran3 enchant pandoc -qq
sudo apt-get install graphviz libgfortran3 enchant pandoc -qq
# doctr and pdr
echo python -m pip install sphinx sphinx-immaterial jupyter nbconvert numpydoc pyyaml doctr pandas-datareader simplegeneric seaborn sphinxcontrib-spelling nbsphinx
python -m pip install sphinx sphinx-immaterial jupyter nbconvert numpydoc pyyaml doctr pandas-datareader simplegeneric seaborn sphinxcontrib-spelling nbsphinx

# Add pymc3 and theano for statespace_sarimax_pymc3.
echo conda install theano pymc3 -y
conda install theano pymc3 -y
