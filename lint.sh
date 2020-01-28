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
        examples/ \
        setup.py \
        statsmodels/__init__.py \
        statsmodels/_version.py \
        statsmodels/api.py \
        statsmodels/base/__init__.py \
        statsmodels/base/distributed_estimation.py \
        statsmodels/base/elastic_net.py \
        statsmodels/base/tests/test_data.py \
        statsmodels/base/tests/test_distributed_estimation.py \
        statsmodels/base/tests/test_shrink_pickle.py \
        statsmodels/base/tests/test_transform.py \
        statsmodels/compat/ \
        statsmodels/conftest.py \
        statsmodels/datasets/__init__.py \
        statsmodels/datasets/anes96/__init__.py \
        statsmodels/datasets/cancer/__init__.py \
        statsmodels/datasets/ccard/__init__.py \
        statsmodels/datasets/china_smoking/__init__.py \
        statsmodels/datasets/co2/__init__.py \
        statsmodels/datasets/committee/__init__.py \
        statsmodels/datasets/copper/__init__.py \
        statsmodels/datasets/cpunish/__init__.py \
        statsmodels/datasets/elec_equip/ \
        statsmodels/datasets/elnino/__init__.py \
        statsmodels/datasets/engel/__init__.py \
        statsmodels/datasets/fair/__init__.py \
        statsmodels/datasets/fertility/__init__.py \
        statsmodels/datasets/grunfeld/__init__.py \
        statsmodels/datasets/heart/__init__.py \
        statsmodels/datasets/interest_inflation/__init__.py \
        statsmodels/datasets/longley/__init__.py \
        statsmodels/datasets/macrodata/__init__.py \
        statsmodels/datasets/modechoice/__init__.py \
        statsmodels/datasets/nile/__init__.py \
        statsmodels/datasets/randhie/__init__.py \
        statsmodels/datasets/scotland/__init__.py \
        statsmodels/datasets/spector/__init__.py \
        statsmodels/datasets/stackloss/__init__.py \
        statsmodels/datasets/star98/__init__.py \
        statsmodels/datasets/statecrime/__init__.py \
        statsmodels/datasets/strikes/__init__.py \
        statsmodels/datasets/sunspots/__init__.py \
        statsmodels/datasets/tests/ \
        statsmodels/discrete/__init__.py \
        statsmodels/discrete/tests/results/ \
        statsmodels/discrete/tests/test_constrained.py \
        statsmodels/distributions/__init__.py \
        statsmodels/duration/__init__.py \
        statsmodels/duration/_kernel_estimates.py \
        statsmodels/duration/api.py \
        statsmodels/duration/tests/results/ \
        statsmodels/duration/tests/test_survfunc.py \
        statsmodels/emplike/__init__.py \
        statsmodels/emplike/api.py \
        statsmodels/emplike/tests/ \
        statsmodels/examples/tests/ \
        statsmodels/formula/ \
        statsmodels/gam/ \
        statsmodels/genmod/__init__.py \
        statsmodels/genmod/_tweedie_compound_poisson.py \
        statsmodels/genmod/api.py \
        statsmodels/genmod/bayes_mixed_glm.py \
        statsmodels/genmod/families/ \
        statsmodels/genmod/generalized_estimating_equations.py \
        statsmodels/genmod/qif.py \
        statsmodels/genmod/tests/results/ \
        statsmodels/genmod/tests/test_gee.py \
        statsmodels/genmod/tests/test_qif.py \
        statsmodels/graphics/__init__.py \
        statsmodels/graphics/api.py \
        statsmodels/graphics/functional.py \
        statsmodels/graphics/tests/test_agreement.py \
        statsmodels/graphics/tests/test_boxplots.py \
        statsmodels/graphics/tests/test_correlation.py \
        statsmodels/graphics/tests/test_functional.py \
        statsmodels/graphics/tests/test_gofplots.py \
        statsmodels/graphics/tsaplots.py \
        statsmodels/imputation/__init__.py \
        statsmodels/interface/ \
        statsmodels/iolib/__init__.py \
        statsmodels/iolib/api.py \
        statsmodels/iolib/openfile.py \
        statsmodels/iolib/smpickle.py \
        statsmodels/iolib/summary2.py \
        statsmodels/iolib/table.py \
        statsmodels/iolib/tableformatting.py \
        statsmodels/iolib/tests/results/ \
        statsmodels/iolib/tests/test_pickle.py \
        statsmodels/miscmodels/__init__.py \
        statsmodels/miscmodels/tests/test_tarma.py \
        statsmodels/multivariate/__init__.py \
        statsmodels/multivariate/api.py \
        statsmodels/multivariate/factor_rotation/_analytic_rotation.py \
        statsmodels/multivariate/pca.py \
        statsmodels/multivariate/tests/results/ \
        statsmodels/nonparametric/__init__.py \
        statsmodels/nonparametric/api.py \
        statsmodels/nonparametric/tests/results/ \
        statsmodels/regression/__init__.py \
        statsmodels/regression/_prediction.py \
        statsmodels/regression/_tools.py \
        statsmodels/regression/dimred.py \
        statsmodels/regression/mixed_linear_model.py \
        statsmodels/regression/process_regression.py \
        statsmodels/regression/recursive_ls.py \
        statsmodels/regression/rolling.py \
        statsmodels/regression/tests/results/ \
        statsmodels/regression/tests/test_dimred.py \
        statsmodels/regression/tests/test_lme.py \
        statsmodels/regression/tests/test_processreg.py \
        statsmodels/regression/tests/test_quantile_regression.py \
        statsmodels/regression/tests/test_rolling.py \
        statsmodels/regression/tests/test_tools.py \
        statsmodels/resampling/ \
        statsmodels/robust/ \
        statsmodels/sandbox/__init__.py \
        statsmodels/sandbox/distributions/__init__.py \
        statsmodels/sandbox/distributions/tests/test_gof_new.py \
        statsmodels/sandbox/distributions/try_pot.py \
        statsmodels/sandbox/nonparametric/__init__.py \
        statsmodels/sandbox/panel/correlation_structures.py \
        statsmodels/sandbox/regression/tests/results_gmm_griliches.py \
        statsmodels/sandbox/regression/tests/results_gmm_griliches_iter.py \
        statsmodels/sandbox/regression/tests/results_gmm_poisson.py \
        statsmodels/sandbox/regression/tests/results_ivreg2_griliches.py \
        statsmodels/sandbox/stats/__init__.py \
        statsmodels/sandbox/stats/ex_multicomp.py \
        statsmodels/sandbox/stats/tests/ \
        statsmodels/src/ \
        statsmodels/stats/__init__.py \
        statsmodels/stats/_knockoff.py \
        statsmodels/stats/_lilliefors.py \
        statsmodels/stats/_lilliefors_critical_values.py \
        statsmodels/stats/api.py \
        statsmodels/stats/base.py \
        statsmodels/stats/contingency_tables.py \
        statsmodels/stats/correlation_tools.py \
        statsmodels/stats/diagnostic.py \
        statsmodels/stats/dist_dependence_measures.py \
        statsmodels/stats/knockoff_regeffects.py \
        statsmodels/stats/libqsturng/__init__.py \
        statsmodels/stats/moment_helpers.py \
        statsmodels/stats/multicomp.py \
        statsmodels/stats/oaxaca.py \
        statsmodels/stats/regularized_covariance.py \
        statsmodels/stats/stattools.py \
        statsmodels/stats/tabledist.py \
        statsmodels/stats/tests/results/ \
        statsmodels/stats/tests/test_anova_rm.py \
        statsmodels/stats/tests/test_correlation.py \
        statsmodels/stats/tests/test_descriptivestats.py \
        statsmodels/stats/tests/test_diagnostic.py \
        statsmodels/stats/tests/test_dist_dependant_measures.py \
        statsmodels/stats/tests/test_knockoff.py \
        statsmodels/stats/tests/test_lilliefors.py \
        statsmodels/stats/tests/test_moment_helpers.py \
        statsmodels/stats/tests/test_multi.py \
        statsmodels/stats/tests/test_oaxaca.py \
        statsmodels/stats/tests/test_outliers_influence.py \
        statsmodels/stats/tests/test_qsturng.py \
        statsmodels/stats/tests/test_regularized_covariance.py \
        statsmodels/stats/tests/test_tabledist.py \
        statsmodels/tests/ \
        statsmodels/tools/decorators.py \
        statsmodels/tools/docstring.py \
        statsmodels/tools/linalg.py \
        statsmodels/tools/sm_exceptions.py \
        statsmodels/tools/tests/test_decorators.py \
        statsmodels/tools/tests/test_docstring.py \
        statsmodels/tools/tests/test_linalg.py \
        statsmodels/tools/validation/ \
        statsmodels/tools/web.py \
        statsmodels/tsa/__init__.py \
        statsmodels/tsa/_bds.py \
        statsmodels/tsa/adfvalues.py \
        statsmodels/tsa/api.py \
        statsmodels/tsa/ar_model.py \
        statsmodels/tsa/arima/ \
        statsmodels/tsa/arima_model.py \
        statsmodels/tsa/base/__init__.py \
        statsmodels/tsa/base/tests/test_datetools.py \
        statsmodels/tsa/exponential_smoothing/ \
        statsmodels/tsa/filters/__init__.py \
        statsmodels/tsa/filters/api.py \
        statsmodels/tsa/filters/bk_filter.py \
        statsmodels/tsa/filters/hp_filter.py \
        statsmodels/tsa/filters/tests/ \
        statsmodels/tsa/innovations/ \
        statsmodels/tsa/interp/__init__.py \
        statsmodels/tsa/kalmanf/ \
        statsmodels/tsa/regime_switching/ \
        statsmodels/tsa/seasonal.py \
        statsmodels/tsa/statespace/ \
        statsmodels/tsa/tests/results/ \
        statsmodels/tsa/tests/test_ar.py \
        statsmodels/tsa/tests/test_arima.py \
        statsmodels/tsa/tests/test_bds.py \
        statsmodels/tsa/tests/test_seasonal.py \
        statsmodels/tsa/tests/test_stl.py \
        statsmodels/tsa/tests/test_x13.py \
        statsmodels/tsa/vector_ar/__init__.py \
        statsmodels/tsa/vector_ar/api.py \
        statsmodels/tsa/vector_ar/dynamic.py \
        statsmodels/tsa/vector_ar/hypothesis_test_results.py \
        statsmodels/tsa/vector_ar/tests/JMulTi_results/ \
        statsmodels/tsa/vector_ar/tests/Matlab_results/ \
        statsmodels/tsa/vector_ar/tests/results/ \
        statsmodels/tsa/x13.py \
        tools/
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
