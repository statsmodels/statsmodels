#!/bin/bash
cd ../..

## declare Python and Numpy Versions
declare -a PY_VERSIONS=( "27" "33" "34" )
declare -a NPY_VERSIONS=( "18" "19" )

## Loop across Python and Numpy
for PY in "${PY_VERSIONS[@]}"
do
    export CONDA_PY=$PY
    for NPY in "${NPY_VERSIONS[@]}"
    do
        export CONDA_NPY=$NPY
        binstar remove statsmodels/statsmodels/0.6.0_dev/linux-64/statsmodels-0.6.0_dev-np${NPY}py${PY}_0.tar.bz2 -f
        conda build ./tools/binstar
    done
done
