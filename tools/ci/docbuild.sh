# Change to doc directory
cd "$SRCDIR"/docs

set -e
# Clean up
echo '================================= Clean ================================='
make clean
git clean -xdf

# Build documentation
echo '========================================================================'
echo '=                        Building documentation                        ='
echo '========================================================================'
echo 'make html 2>&1 | tee doc_build.log'
# Multithreaded build
export O="-j auto"
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
  echo "contents of build"
  ls build
  echo "contents of build/html"
  ls build/html
  exit 1;
fi;


# Deploy with doctr
cd "$SRCDIR"
if [[ -z "$TRAVIS_TAG" ]]; then
  doctr deploy --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io devel > /dev/null;
else
  if [[ "$TRAVIS_TAG" != *"dev"* ]]; then  # do not push on dev tags
    echo doctr deploy --build-tags --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io "$TRAVIS_TAG" > /dev/null;
    doctr deploy --build-tags --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io "$TRAVIS_TAG" > /dev/null;
    if [[ "$TRAVIS_TAG" != *"rc"* ]]; then  # do not push on main on rc
      echo doctr deploy --build-tags --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io stable > /dev/null;
      doctr deploy --build-tags --built-docs docs/build/html/ --deploy-repo statsmodels/statsmodels.github.io stable > /dev/null;
    fi;
  fi;
fi;

echo Finished...