@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)

set SOURCEDIR=source
set BUILDDIR=build
set SPHINXPROJ=statsmodels
set SPHINXOPTS=-j auto

set TOOLSPATH=../tools
set DATASETBUILD=dataset_rst.py
set NOTEBOOKBUILD=nbgenerate.py
set FOLDTOC=fold_toc.py

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

if "%1" == "html" (
    echo mkdir %BUILDDIR%\html\_static
    mkdir %BUILDDIR%\html\_static
	echo python %TOOLSPATH%/%NOTEBOOKBUILD% --parallel --report-errors --skip-existing
    python %TOOLSPATH%/%NOTEBOOKBUILD% --parallel --report-errors --skip-existing
    echo python %TOOLSPATH%/%DATASETBUILD%
    python %TOOLSPATH%/%DATASETBUILD%
)

echo %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
if errorlevel 1 exit /b 1

if "%1" == "html" (
    echo xcopy /s /y source\examples\notebooks\generated\*.html %BUILDDIR%\html\examples\notebooks\generated\*.html
    xcopy /s /y source\examples\notebooks\generated\*.html %BUILDDIR%\html\examples\notebooks\generated\*.html
	echo python %TOOLSPATH%/%FOLDTOC% %BUILDDIR%/html/index.html
	python %TOOLSPATH%/%FOLDTOC% %BUILDDIR%/html/index.html
	echo python %TOOLSPATH%/%FOLDTOC% %BUILDDIR%/html/examples/index.html ../_static
	python %TOOLSPATH%/%FOLDTOC% %BUILDDIR%/html/examples/index.html ../_static
	echo python %TOOLSPATH%/%FOLDTOC% %BUILDDIR%/html/dev/index.html ../_static
	python %TOOLSPATH%/%FOLDTOC% %BUILDDIR%/html/dev/index.html ../_static
    if NOT EXIST %BUILDDIR%/html/examples/notebooks/generated mkdir %BUILDDIR%\html\examples\notebooks\generated
)

goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
