# Script to text examples and build docs on travis

# Change to doc directory
cd ${SRCDIR}/docs

# Run notebooks as tests
pytest ../statsmodels/examples/tests

# Clean up
echo '================================= Clean ================================='
make clean
git clean -xdf

# Build documentation
echo '========================================================================'
echo '=                        Building documentation                        ='
echo '========================================================================'
echo 'make html > doc_build.log 2>&1'
make html 2>&1 | tee doc_build.log

# Info
echo '========================================================================'
echo '=                 Opportunities To Improve (Warnings)                  ='
echo '========================================================================'
cat doc_build.log | grep -E '(WARNING)' | grep -v '(noindex|toctree)'

# Check log
echo '========================================================================'
echo '=          Broken Behavior (Errors and Warnings to be Fixed)           ='
echo '========================================================================'
cat doc_build.log | grep -E '(SEVERE|ERROR|WARNING)' | grep -Ev '(noindex|toctree)'

# Deploy with doctr
cd ${SRCDIR};
if [[ -z "$TRAVIS_TAG" ]]; then
  doctr deploy --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io devel;
else
  doctr deploy --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io "$TRAVIS_TAG";
  doctr deploy --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io stable;
fi;
