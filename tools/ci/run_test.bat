@echo on

@IF not Defined PYTEST_DIRECTIVES GOTO :RUN_FROM_INSTALL

@echo Running test from %CD%
@echo pytest -n auto -r s statsmodels --dist loadscope --skip-examples %PYTEST_DIRECTIVES%
pytest -n auto -r s statsmodels --dist loadscope --skip-examples %PYTEST_DIRECTIVES%

@GOTO :End

:RUN_FROM_INSTALL

mkdir test-run
cd test-run
@echo PYTEST_DIRECTIVES is not defined, testing using install
@echo Running test from %CD%
@echo python -c "import statsmodels;statsmodels.test(['-r s', '-n 2', '--skip-examples'], exit=True)
python -c "import statsmodels;statsmodels.test(['-r s', '-n 2', '--skip-examples'], exit=True)

:End
