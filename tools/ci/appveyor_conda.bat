REM Install required conda version and libraries
@echo on
curl -fsS -o c:\Miniconda.exe "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-%PYTHON_ARCH%.exe"
START /WAIT C:\Miniconda.exe /S /D=C:\Py
set PATH=C:\Py;C:\Py\Scripts;C:\Py\Library\bin;%PATH%
conda config --set always_yes yes
conda update conda --quiet
IF Defined SCIPY (
    conda install numpy=%NUMPY% scipy=%SCIPY% icc_rt cython pandas pip nose patsy --quiet
) else (
    conda install numpy scipy cython pandas pip nose patsy --quiet
)
