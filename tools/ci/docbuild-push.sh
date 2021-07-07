#!/usr/bin/env bash

pushd statsmodels.github.io
echo "Change the remote"
git remote set-url origin https://${PERSONAL_ACCESS_TOKEN}@github.com/statsmodels/statsmodels.github.io.git
echo "Pushing"
git push
popd

