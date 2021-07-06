#!/usr/bin/env bash

echo "Set git email and name"
git config --global user.email "bot@statsmodels.org"
git config --global user.name "Statsmodels Doc Bot"

echo "Remove devel"
rm -rf statsmodels.github.io/devel
echo "Make a new devel"
mkdir statsmodels.github.io/devel
echo "Checking for non-dev tagged build"
if [[ -n "${GIT_TAG}" ]] && [[ "${GIT_TAG}" != *dev ]]; then
  echo "Tag ${GIT_TAG} is defined"
  echo "Copy docs tag"
  echo mkdir statsmodels.github.io/"${GIT_TAG}"
  mkdir statsmodels.github.io/"${GIT_TAG}"
  echo cp -R "${PWD}"/docs/build/html/* statsmodels.github.io/"${GIT_TAG}"
  cp -R "${PWD}"/docs/build/html/* statsmodels.github.io/"${GIT_TAG}"
  pushd statsmodels.github.io || exit
  echo git add -A "${GIT_TAG}"/.
  git add -A "${GIT_TAG}"/.
  popd || exit
  # Also copy to stable
  rm -rf statsmodels.github.io/stable
  echo mkdir statsmodels.github.io/stable
  mkdir statsmodels.github.io/stable
  echo cp -R "${PWD}"/docs/build/html/* statsmodels.github.io/stable
  cp -R "${PWD}"/docs/build/html/* statsmodels.github.io/stable
  pushd statsmodels.github.io || exit
  echo git add -A stable/.
  git add -A stable/.
  popd || exit
else
  echo "Tag is ${GIT_TAG}. Not updating fixed documents"
fi
echo "Copy docs to devel"
echo cp -R "${PWD}"/docs/build/html/* statsmodels.github.io/devel
cp -R "${PWD}"/docs/build/html/* statsmodels.github.io/devel
echo "Add devel"
pushd statsmodels.github.io || exit
echo git add -A devel/\*
git add -A devel/\*
echo "Change remote"
# TODO: Enable
# git remote set-url origin https://USERNAME:"${GH_PAGES_TOKEN}"@github.com/statsmodels/statsmodels.github.io.git
echo "Github Actions doc build after commit ${GITHUB_SHA::8}"
git commit -a -m "Github Actions doc build after commit ${GITHUB_SHA::8}"
echo "Push"
# TODO: Enable
# git push -f
popd || exit
