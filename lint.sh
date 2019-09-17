#!/bin/bash

echo "inside $0"

RET=0

if [ "$LINT" == true ]; then
    echo "Linting all files with limited rules"
    flake8 statsmodels
    if [ $? -ne "0" ]; then
        echo "Changed files failed linting using the required set of rules."
        echo "Additions and changes must conform to Python code style rules."
        RET=1
    fi

    # Run with --isolated to ignore config files, the files included here
    # pass _all_ flake8 checks
    echo "Linting known clean files with strict rules"
    flake8 --isolated \
        statsmodels/__init__.py \
        statsmodels/base/distributed_estimation.py \
        statsmodels/base/elastic_net.py \
        statsmodels/base/tests/test_data.py \
        statsmodels/base/tests/test_distributed_estimation.py \
        statsmodels/base/tests/test_transform.py \
        statsmodels/compat/ \
        statsmodels/conftest.py \
        statsmodels/datasets/elec_equip \
        statsmodels/datasets/tests/ \
        statsmodels/discrete/tests/results/ \
        statsmodels/discrete/tests/test_constrained.py \
        statsmodels/duration/__init__.py \
        statsmodels/duration/tests/results/ \
        statsmodels/emplike/tests/ \
        statsmodels/examples/tests/ \
        statsmodels/formula/ \
        statsmodels/gam/ \
        statsmodels/genmod/_tweedie_compound_poisson.py \
        statsmodels/genmod/bayes_mixed_glm.py \
        statsmodels/genmod/families/ \
        statsmodels/genmod/generalized_estimating_equations.py \
        statsmodels/genmod/qif.py \
        statsmodels/genmod/tests/results/ \
        statsmodels/genmod/tests/test_gee.py \
        statsmodels/genmod/tests/test_qif.py \
        statsmodels/graphics/tsaplots.py \
        statsmodels/interface/ \
        statsmodels/iolib/openfile.py \
        statsmodels/iolib/smpickle.py \
        statsmodels/iolib/summary2.py \
        statsmodels/iolib/table.py \
        statsmodels/iolib/tableformatting.py \
        statsmodels/iolib/tests/results/ \
        statsmodels/iolib/tests/test_pickle.py \
        statsmodels/multivariate/pca.py \
        statsmodels/multivariate/tests/results/ \
        statsmodels/regression/dimred.py \
        statsmodels/regression/mixed_linear_model.py \
        statsmodels/regression/process_regression.py \
        statsmodels/regression/recursive_ls.py \
        statsmodels/regression/tests/results/ \
        statsmodels/regression/tests/test_dimred.py \
        statsmodels/regression/tests/test_lme.py \
        statsmodels/regression/tests/test_processreg.py \
        statsmodels/regression/tests/test_quantile_regression.py \
        statsmodels/regression/tests/test_tools.py \
        statsmodels/regression/tests/results/ \
        statsmodels/robust/norms.py \
        statsmodels/robust/scale.py \
        statsmodels/resampling/ \
        statsmodels/robust/tests/ \
        statsmodels/sandbox/distributions/tests/test_gof_new.py \
        statsmodels/sandbox/distributions/try_pot.py \
        statsmodels/sandbox/panel/correlation_structures.py \
        statsmodels/sandbox/regression/tests/results_gmm_griliches.py \
        statsmodels/sandbox/regression/tests/results_gmm_griliches_iter.py \
        statsmodels/sandbox/regression/tests/results_gmm_poisson.py \
        statsmodels/sandbox/regression/tests/results_ivreg2_griliches.py \
        statsmodels/sandbox/stats/ex_multicomp.py \
        statsmodels/sandbox/stats/tests/test_multicomp.py \
        statsmodels/stats/__init__.py \
        statsmodels/stats/_knockoff.py \
        statsmodels/stats/base.py \
        statsmodels/stats/correlation_tools.py \
        statsmodels/stats/knockoff_regeffects.py \
        statsmodels/stats/moment_helpers.py \
        statsmodels/stats/multicomp.py \
        statsmodels/stats/regularized_covariance.py \
        statsmodels/stats/stattools.py \
        statsmodels/stats/oaxaca.py \
        statsmodels/stats/tests/results/ \
        statsmodels/stats/tests/test_anova_rm.py \
        statsmodels/stats/tests/test_correlation.py \
        statsmodels/stats/tests/test_descriptivestats.py \
        statsmodels/stats/tests/test_knockoff.py \
        statsmodels/stats/tests/test_lilliefors.py \
        statsmodels/stats/tests/test_moment_helpers.py \
        statsmodels/stats/tests/test_multi.py \
        statsmodels/stats/tests/test_qsturng.py \
        statsmodels/stats/tests/test_regularized_covariance.py \
        statsmodels/stats/tests/test_oaxaca.py \
        statsmodels/tests/ \
        statsmodels/tools/decorators.py \
        statsmodels/tools/docstring.py \
        statsmodels/tools/linalg.py \
        statsmodels/tools/sm_exceptions.py \
        statsmodels/tools/tests/test_decorators.py \
        statsmodels/tools/tests/test_linalg.py \
        statsmodels/tools/validation/ \
        statsmodels/tools/web.py \
        statsmodels/tools/tests/test_docstring.py \
        statsmodels/tsa/adfvalues.py \
        statsmodels/tsa/ar_model.py \
        statsmodels/tsa/arima/ \
        statsmodels/tsa/arima_model.py \
        statsmodels/tsa/base/tests/test_datetools.py \
        statsmodels/tsa/filters/tests/ \
        statsmodels/tsa/innovations/ \
        statsmodels/tsa/kalmanf/ \
        statsmodels/tsa/regime_switching/ \
        statsmodels/tsa/seasonal.py \
        statsmodels/tsa/statespace/ \
        statsmodels/tsa/tests/results/ \
        statsmodels/tsa/tests/test_ar.py \
        statsmodels/tsa/tests/test_arima.py \
        statsmodels/tsa/tests/test_seasonal.py \
        statsmodels/tsa/tests/test_stl.py \
        statsmodels/tsa/tests/test_x13.py \
        statsmodels/tsa/vector_ar/dynamic.py \
        statsmodels/tsa/vector_ar/tests/results/ \
        statsmodels/tsa/x13.py \
        examples/ \
        tools/ \
        setup.py
    if [ $? -ne "0" ]; then
        echo "Previously passing files failed linting."
        RET=1
    fi

    # Tests any new python files
    if [ -f $(git rev-parse --git-dir)/shallow ]; then
        # Unshallow only when required, i.e., on CI
        echo "Repository is shallow"
        git fetch --unshallow --quiet
    fi
    git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
    git fetch origin --quiet
    NEW_FILES=$(git diff origin/master --name-status -u -- "*.py" | grep ^A | cut -c 3- | paste -sd " " -)
    if [ -n "$NEW_FILES" ]; then
        echo "Linting newly added files with strict rules"
        echo "New files: $NEW_FILES"
        flake8 --isolated $(eval echo $NEW_FILES)
        if [ $? -ne "0" ]; then
            echo "New files failed linting."
            RET=1
        fi
    else
        echo "No new files to lint"
    fi
fi

exit "$RET"
