# Script to text examples and build docs on travis
echo '========================================================================'
echo '=                        Checking Doc Strings                          ='
echo '========================================================================'

cd "$SRCDIR"
python "$SRCDIR"/tools/validate_docstrings.py --errors=GL03,GL04,GL05,GL06,GL07,GL09,SS04,SS05,PR03,PR04,PR05,PR10,EX04,RT01,RT04,RT05,SA05

# Change to doc directory
cd "$SRCDIR"/docs

# Run notebooks as tests
pytest ../statsmodels/examples/tests

set -e
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

set +e
# Info
echo '========================================================================'
echo '=                 Opportunities To Improve (Warnings)                  ='
echo '========================================================================'
grep -E '(WARNING)' doc_build.log | grep -v '(noindex|toctree)'

# Check log
echo '========================================================================'
echo '=          Broken Behavior (Errors and Warnings to be Fixed)           ='
echo '========================================================================'
grep -E '(SEVERE|ERROR|WARNING)' doc_build.log | grep -Ev '(noindex|toctree)'

# Check that docbuild succeeded. Exit 1 if not.
echo "Checking if index.html exists"
if [ -f "build/html/index.html" ]; then
  echo "docbuild succeeded.";
else
  echo "docbuild failed. Aborting doctr.";
  exit 1;
fi;

# Build documentation
echo '========================================================================'
echo '=                        Checking Spelling                             ='
echo '========================================================================'
echo 'make spelling > doc_spelling.log 2>&1'
make spelling
if [ -f "build/spelling/output.txt" ]; then
  cat build/spelling/output.txt
fi;


# Deploy with doctr
cd "$SRCDIR"
if [[ -z "$TRAVIS_TAG" ]]; then
  doctr deploy --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io devel;
else
  if [[ "$TRAVIS_TAG" != *"dev"* ]]; then  # do not push on dev tags
    doctr deploy --build-tags --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io "$TRAVIS_TAG";
    doctr deploy --build-tags --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io stable;
  fi;
fi;
