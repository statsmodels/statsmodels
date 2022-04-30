REM Install packages using pip
set PATH=%PYTHON%;%PYTHON%\Scripts;%PATH%
python -m pip install -U pip
IF Defined SCIPY (
    python -m pip install numpy scipy==%SCIPY% "cython>=0.29.28,<3.0.0" pandas nose patsy
) else (
    python -m pip install numpy scipy "cython>=0.29.28,<3.0.0" pandas nose patsy
)
