#!/bin/bash

echo "inside $0"

RET=0

if [ "$LINT" == true ]; then
    echo "Running ruff check"
    ruff check statsmodels

    echo "Running flake8 linting"
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
    # Default flake8 rules plus the additional rules from setup.cfg
    flake8 --isolated  \
        --max-line-length 88 \
        --ignore=E121,E123,E126,E226,E24,E704,W503,W504,E741,E203 \
        examples \
        setup.py \
        statsmodels/__init__.py \
        statsmodels/api.py \
        statsmodels/base/__init__.py \
        statsmodels/base/distributed_estimation.py \
        statsmodels/base/elastic_net.py \
        statsmodels/base/tests/__init__.py \
        statsmodels/base/tests/test_data.py \
        statsmodels/base/tests/test_distributed_estimation.py \
        statsmodels/base/tests/test_optimize.py \
        statsmodels/base/tests/test_shrink_pickle.py \
        statsmodels/base/tests/test_transform.py \
        statsmodels/compat \
        statsmodels/compat/tests \
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
        statsmodels/datasets/danish_data \
        statsmodels/datasets/elec_equip \
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
        statsmodels/datasets/tests \
        statsmodels/discrete/__init__.py \
        statsmodels/discrete/diagnostic.py \
        statsmodels/discrete/tests/__init__.py \
        statsmodels/discrete/tests/results \
        statsmodels/discrete/tests/test_constrained.py \
        statsmodels/discrete/tests/test_truncated_model.py \
        statsmodels/discrete/truncated_model.py \
        statsmodels/distributions/__init__.py \
        statsmodels/distributions/bernstein.py \
        statsmodels/distributions/empirical_distribution.py \
        statsmodels/distributions/copula/__init__.py \
        statsmodels/distributions/copula/api.py \
        statsmodels/distributions/copula/archimedean.py \
        statsmodels/distributions/copula/copulas.py \
        statsmodels/distributions/copula/depfunc_ev.py \
        statsmodels/distributions/copula/elliptical.py \
        statsmodels/distributions/copula/extreme_value.py \
        statsmodels/distributions/copula/other_copulas.py \
        statsmodels/distributions/copula/tests/test_model.py \
        statsmodels/distributions/copula/transforms.py \
        statsmodels/distributions/tests/__init__.py \
        statsmodels/distributions/tests/test_bernstein.py \
        statsmodels/distributions/tests/test_discrete.py \
        statsmodels/distributions/tests/test_ecdf.py \
        statsmodels/distributions/tests/test_tools.py \
        statsmodels/duration/__init__.py \
        statsmodels/duration/_kernel_estimates.py \
        statsmodels/duration/api.py \
        statsmodels/duration/tests/__init__.py \
        statsmodels/duration/tests/results \
        statsmodels/duration/tests/test_survfunc.py \
        statsmodels/emplike/__init__.py \
        statsmodels/emplike/api.py \
        statsmodels/emplike/tests \
        statsmodels/emplike/tests/results \
        statsmodels/examples/ex_ordered_model.py \
        statsmodels/examples/tests \
        statsmodels/examples/tsa/ex_var_reorder.py \
        statsmodels/formula \
        statsmodels/formula/tests \
        statsmodels/gam \
        statsmodels/gam/gam_cross_validation \
        statsmodels/gam/tests \
        statsmodels/gam/tests/results \
        statsmodels/genmod/__init__.py \
        statsmodels/genmod/_tweedie_compound_poisson.py \
        statsmodels/genmod/api.py \
        statsmodels/genmod/bayes_mixed_glm.py \
        statsmodels/genmod/families \
        statsmodels/genmod/families/tests \
        statsmodels/genmod/generalized_estimating_equations.py \
        statsmodels/genmod/qif.py \
        statsmodels/genmod/tests/__init__.py \
        statsmodels/genmod/tests/results \
        statsmodels/genmod/tests/test_gee.py \
        statsmodels/genmod/tests/test_qif.py \
        statsmodels/graphics/__init__.py \
        statsmodels/graphics/api.py \
        statsmodels/graphics/functional.py \
        statsmodels/graphics/tests/__init__.py \
        statsmodels/graphics/tests/test_agreement.py \
        statsmodels/graphics/tests/test_boxplots.py \
        statsmodels/graphics/tests/test_correlation.py \
        statsmodels/graphics/tests/test_functional.py \
        statsmodels/graphics/tests/test_gofplots.py \
        statsmodels/graphics/tests/test_tsaplots.py \
        statsmodels/graphics/tsaplots.py \
        statsmodels/imputation/__init__.py \
        statsmodels/imputation/tests/__init__.py \
        statsmodels/interface \
        statsmodels/iolib/__init__.py \
        statsmodels/iolib/api.py \
        statsmodels/iolib/openfile.py \
        statsmodels/iolib/smpickle.py \
        statsmodels/iolib/summary2.py \
        statsmodels/iolib/table.py \
        statsmodels/iolib/tableformatting.py \
        statsmodels/iolib/tests/__init__.py \
        statsmodels/iolib/tests/results \
        statsmodels/iolib/tests/test_pickle.py \
        statsmodels/iolib/tests/test_summary2.py \
        statsmodels/miscmodels/__init__.py \
        statsmodels/miscmodels/tests/__init__.py \
        statsmodels/miscmodels/tests/results \
        statsmodels/multivariate/__init__.py \
        statsmodels/multivariate/api.py \
        statsmodels/multivariate/factor_rotation/_analytic_rotation.py \
        statsmodels/multivariate/factor_rotation/tests/__init__.py \
        statsmodels/multivariate/pca.py \
        statsmodels/multivariate/tests/__init__.py \
        statsmodels/multivariate/tests/results \
        statsmodels/nonparametric/__init__.py \
        statsmodels/nonparametric/api.py \
        statsmodels/nonparametric/kde.py \
        statsmodels/nonparametric/kernels_asymmetric.py \
        statsmodels/nonparametric/tests/__init__.py \
        statsmodels/nonparametric/tests/results \
        statsmodels/nonparametric/tests/test_asymmetric.py \
        statsmodels/othermod \
        statsmodels/othermod/tests \
        statsmodels/othermod/tests/results \
        statsmodels/regression/__init__.py \
        statsmodels/regression/_prediction.py \
        statsmodels/regression/_tools.py \
        statsmodels/regression/dimred.py \
        statsmodels/regression/mixed_linear_model.py \
        statsmodels/regression/process_regression.py \
        statsmodels/regression/recursive_ls.py \
        statsmodels/regression/rolling.py \
        statsmodels/regression/tests/__init__.py \
        statsmodels/regression/tests/results \
        statsmodels/regression/tests/test_dimred.py \
        statsmodels/regression/tests/test_lme.py \
        statsmodels/regression/tests/test_processreg.py \
        statsmodels/regression/tests/test_quantile_regression.py \
        statsmodels/regression/tests/test_rolling.py \
        statsmodels/regression/tests/test_tools.py \
        statsmodels/robust \
        statsmodels/robust/tests \
        statsmodels/robust/tests/results \
        statsmodels/sandbox/__init__.py \
        statsmodels/sandbox/archive/__init__.py \
        statsmodels/sandbox/distributions/__init__.py \
        statsmodels/sandbox/distributions/examples/__init__.py \
        statsmodels/sandbox/distributions/tests/__init__.py \
        statsmodels/sandbox/distributions/tests/test_gof_new.py \
        statsmodels/sandbox/distributions/try_pot.py \
        statsmodels/sandbox/mcevaluate/__init__.py \
        statsmodels/sandbox/nonparametric/__init__.py \
        statsmodels/sandbox/nonparametric/tests/__init__.py \
        statsmodels/sandbox/panel/__init__.py \
        statsmodels/sandbox/panel/correlation_structures.py \
        statsmodels/sandbox/panel/tests/__init__.py \
        statsmodels/sandbox/regression/tests/__init__.py \
        statsmodels/sandbox/regression/tests/results_gmm_griliches.py \
        statsmodels/sandbox/regression/tests/results_gmm_griliches_iter.py \
        statsmodels/sandbox/regression/tests/results_gmm_poisson.py \
        statsmodels/sandbox/regression/tests/results_ivreg2_griliches.py \
        statsmodels/sandbox/stats/__init__.py \
        statsmodels/sandbox/stats/diagnostic.py \
        statsmodels/sandbox/stats/tests \
        statsmodels/sandbox/tests/__init__.py \
        statsmodels/sandbox/tools/__init__.py \
        statsmodels/src \
        statsmodels/stats/__init__.py \
        statsmodels/stats/_delta_method.py \
        statsmodels/stats/_knockoff.py \
        statsmodels/stats/_lilliefors.py \
        statsmodels/stats/_lilliefors_critical_values.py \
        statsmodels/stats/api.py \
        statsmodels/stats/base.py \
        statsmodels/stats/contingency_tables.py \
        statsmodels/stats/correlation_tools.py \
        statsmodels/stats/diagnostic.py \
        statsmodels/stats/diagnostic_gen.py \
        statsmodels/stats/dist_dependence_measures.py \
        statsmodels/stats/effect_size.py \
        statsmodels/stats/knockoff_regeffects.py \
        statsmodels/stats/libqsturng/__init__.py \
        statsmodels/stats/libqsturng/tests/__init__.py \
        statsmodels/stats/meta_analysis.py \
        statsmodels/stats/moment_helpers.py \
        statsmodels/stats/multicomp.py \
        statsmodels/stats/multivariate.py \
        statsmodels/stats/nonparametric.py \
        statsmodels/stats/oaxaca.py \
        statsmodels/stats/oneway.py \
        statsmodels/stats/rates.py \
        statsmodels/stats/regularized_covariance.py \
        statsmodels/stats/robust_compare.py \
        statsmodels/stats/stattools.py \
        statsmodels/stats/tabledist.py \
        statsmodels/stats/tests/__init__.py \
        statsmodels/stats/tests/results \
        statsmodels/stats/tests/test_anova_rm.py \
        statsmodels/stats/tests/test_base.py \
        statsmodels/stats/tests/test_correlation.py \
        statsmodels/stats/tests/test_deltacov.py \
        statsmodels/stats/tests/test_descriptivestats.py \
        statsmodels/stats/tests/test_diagnostic.py \
        statsmodels/stats/tests/test_dist_dependant_measures.py \
        statsmodels/stats/tests/test_effectsize.py \
        statsmodels/stats/tests/test_knockoff.py \
        statsmodels/stats/tests/test_lilliefors.py \
        statsmodels/stats/tests/test_meta.py \
        statsmodels/stats/tests/test_moment_helpers.py \
        statsmodels/stats/tests/test_multi.py \
        statsmodels/stats/tests/test_oaxaca.py \
        statsmodels/stats/tests/test_oneway.py \
        statsmodels/stats/tests/test_outliers_influence.py \
        statsmodels/stats/tests/test_qsturng.py \
        statsmodels/stats/tests/test_rates_poisson.py \
        statsmodels/stats/tests/test_regularized_covariance.py \
        statsmodels/stats/tests/test_robust_compare.py \
        statsmodels/stats/tests/test_sandwich.py \
        statsmodels/stats/tests/test_tabledist.py \
        statsmodels/tests \
        statsmodels/tools/__init__.py \
        statsmodels/tools/decorators.py \
        statsmodels/tools/docstring.py \
        statsmodels/tools/eval_measures.py \
        statsmodels/tools/grouputils.py \
        statsmodels/tools/linalg.py \
        statsmodels/tools/parallel.py \
        statsmodels/tools/rng_qrng.py \
        statsmodels/tools/rootfinding.py \
        statsmodels/tools/sequences.py \
        statsmodels/tools/sm_exceptions.py \
        statsmodels/tools/testing.py \
        statsmodels/tools/tests/__init__.py \
        statsmodels/tools/tests/test_data.py \
        statsmodels/tools/tests/test_decorators.py \
        statsmodels/tools/tests/test_docstring.py \
        statsmodels/tools/tests/test_eval_measures.py \
        statsmodels/tools/tests/test_linalg.py \
        statsmodels/tools/tests/test_testing.py \
        statsmodels/tools/tests/test_transform_model.py \
        statsmodels/tools/tests/test_web.py \
        statsmodels/tools/transform_model.py \
        statsmodels/tools/typing.py \
        statsmodels/tools/validation \
        statsmodels/tools/validation/tests \
        statsmodels/tools/web.py \
        statsmodels/tsa/__init__.py \
        statsmodels/tsa/_bds.py \
        statsmodels/tsa/adfvalues.py \
        statsmodels/tsa/api.py \
        statsmodels/tsa/ar_model.py \
        statsmodels/tsa/ardl \
        statsmodels/tsa/ardl/_pss_critical_values \
        statsmodels/tsa/ardl/tests \
        statsmodels/tsa/arima \
        statsmodels/tsa/arima/datasets \
        statsmodels/tsa/arima/datasets/brockwell_davis_2002 \
        statsmodels/tsa/arima/datasets/brockwell_davis_2002/data \
        statsmodels/tsa/arima/estimators \
        statsmodels/tsa/arima/estimators/tests \
        statsmodels/tsa/arima/tests \
        statsmodels/tsa/arima_model.py \
        statsmodels/tsa/arma_mle.py \
        statsmodels/tsa/base/__init__.py \
        statsmodels/tsa/base/prediction.py \
        statsmodels/tsa/base/tests/__init__.py \
        statsmodels/tsa/base/tests/test_datetools.py \
        statsmodels/tsa/base/tests/test_prediction.py \
        statsmodels/tsa/deterministic.py \
        statsmodels/tsa/exponential_smoothing \
        statsmodels/tsa/filters/__init__.py \
        statsmodels/tsa/filters/api.py \
        statsmodels/tsa/filters/bk_filter.py \
        statsmodels/tsa/filters/hp_filter.py \
        statsmodels/tsa/filters/tests \
        statsmodels/tsa/filters/tests/results \
        statsmodels/tsa/forecasting \
        statsmodels/tsa/forecasting/tests \
        statsmodels/tsa/holtwinters \
        statsmodels/tsa/holtwinters/tests \
        statsmodels/tsa/holtwinters/tests/results \
        statsmodels/tsa/innovations \
        statsmodels/tsa/innovations/tests \
        statsmodels/tsa/interp/__init__.py \
        statsmodels/tsa/interp/tests/__init__.py \
        statsmodels/tsa/regime_switching \
        statsmodels/tsa/regime_switching/tests \
        statsmodels/tsa/regime_switching/tests/results \
        statsmodels/tsa/seasonal \
        statsmodels/tsa/statespace \
        statsmodels/tsa/statespace/_filters \
        statsmodels/tsa/statespace/_smoothers \
        statsmodels/tsa/statespace/tests \
        statsmodels/tsa/statespace/tests/results \
        statsmodels/tsa/statespace/tests/results/frbny_nowcast \
        statsmodels/tsa/statespace/tests/results/frbny_nowcast/Nowcasting \
        statsmodels/tsa/statespace/tests/results/frbny_nowcast/Nowcasting/data \
        statsmodels/tsa/statespace/tests/results/frbny_nowcast/Nowcasting/data/US \
        statsmodels/tsa/statespace/tests/results/frbny_nowcast/Nowcasting/functions \
        statsmodels/tsa/stl \
        statsmodels/tsa/stl/tests \
        statsmodels/tsa/tests/__init__.py \
        statsmodels/tsa/tests/results \
        statsmodels/tsa/tests/test_ar.py \
        statsmodels/tsa/tests/test_bds.py \
        statsmodels/tsa/tests/test_deterministic.py \
        statsmodels/tsa/tests/test_seasonal.py \
        statsmodels/tsa/tests/test_tsa_tools.py \
        statsmodels/tsa/tests/test_x13.py \
        statsmodels/tsa/vector_ar/__init__.py \
        statsmodels/tsa/vector_ar/api.py \
        statsmodels/tsa/vector_ar/hypothesis_test_results.py \
        statsmodels/tsa/vector_ar/tests/JMulTi_results \
        statsmodels/tsa/vector_ar/tests/Matlab_results \
        statsmodels/tsa/vector_ar/tests/__init__.py \
        statsmodels/tsa/vector_ar/tests/results \
        statsmodels/tsa/x13.py \
        tools
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
    NEW_FILES=$(git diff origin/main --name-status -u -- "*.py" | grep ^A | cut -c 3- | paste -sd " " -)
    if [ -n "$NEW_FILES" ]; then
        echo "Linting newly added files with strict rules"
        echo "New files: $NEW_FILES"
        flake8 --isolated --max-line-length 88 --ignore=E121,E123,E126,E226,E24,E704,W503,W504,E741,E203 $(eval echo $NEW_FILES)
        if [ $? -ne "0" ]; then
            echo "New files failed linting."
            RET=1
        fi
    else
        echo "No new files to lint"
    fi
fi

exit "$RET"
