@ECHO OFF

REM Command file for Sphinx documentation

set SPHINXBUILD=sphinx-build
set ALLSPHINXOPTS=-d build/doctrees %SPHINXOPTS% source
if NOT "%PAPER%" == "" (
	set ALLSPHINXOPTS=-D latex_paper_size=%PAPER% %ALLSPHINXOPTS%
)

if "%1" == "" goto help

if "%1" == "help" (
	:help
	echo.Please use `make ^<target^>` where ^<target^> is one of
	echo.  html      to make standalone HTML files
	echo.  dirhtml   to make HTML files named index.html in directories
	echo.  pickle    to make pickle files
	echo.  json      to make JSON files
	echo.  htmlhelp  to make HTML files and a HTML help project
	echo.  qthelp    to make HTML files and a qthelp project
	echo.  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter
	echo.  changes   to make an overview over all changed/added/deprecated items
	echo.  linkcheck to check all external links for integrity
	echo.  doctest   to run all doctests embedded in the documentation if enabled
	goto end
)

if "%1" == "clean" (
	for /d %%i in (build\*) do rmdir /q /s %%i
	del /q /s build\*
	goto end
)

if "%1" == "html" (
	%SPHINXBUILD% -b html %ALLSPHINXOPTS% build/html
	echo.
	echo.Build finished. The HTML pages are in build/html.
	goto end
)

if "%1" == "dirhtml" (
	%SPHINXBUILD% -b dirhtml %ALLSPHINXOPTS% build/dirhtml
	echo.
	echo.Build finished. The HTML pages are in build/dirhtml.
	goto end
)

if "%1" == "pickle" (
	%SPHINXBUILD% -b pickle %ALLSPHINXOPTS% build/pickle
	echo.
	echo.Build finished; now you can process the pickle files.
	goto end
)

if "%1" == "json" (
	%SPHINXBUILD% -b json %ALLSPHINXOPTS% build/json
	echo.
	echo.Build finished; now you can process the JSON files.
	goto end
)

if "%1" == "htmlhelp" (
	%SPHINXBUILD% -b htmlhelp %ALLSPHINXOPTS% build/htmlhelp
	echo.
	echo.Build finished; now you can run HTML Help Workshop with the ^
.hhp project file in build/htmlhelp.
	goto end
)

if "%1" == "qthelp" (
	%SPHINXBUILD% -b qthelp %ALLSPHINXOPTS% build/qthelp
	echo.
	echo.Build finished; now you can run "qcollectiongenerator" with the ^
.qhcp project file in build/qthelp, like this:
	echo.^> qcollectiongenerator build\qthelp\statsmodels.qhcp
	echo.To view the help file:
	echo.^> assistant -collectionFile build\qthelp\statsmodels.ghc
	goto end
)

if "%1" == "latex" (
	%SPHINXBUILD% -b latex %ALLSPHINXOPTS% build/latex
	echo.
	echo.Build finished; the LaTeX files are in build/latex.
	goto end
)

if "%1" == "changes" (
	%SPHINXBUILD% -b changes %ALLSPHINXOPTS% build/changes
	echo.
	echo.The overview file is in build/changes.
	goto end
)

if "%1" == "linkcheck" (
	%SPHINXBUILD% -b linkcheck %ALLSPHINXOPTS% build/linkcheck
	echo.
	echo.Link check complete; look for any errors in the above output ^
or in build/linkcheck/output.txt.
	goto end
)

if "%1" == "doctest" (
	%SPHINXBUILD% -b doctest %ALLSPHINXOPTS% build/doctest
	echo.
	echo.Testing of doctests in the sources finished, look at the ^
results in build/doctest/output.txt.
	goto end
)

:end
