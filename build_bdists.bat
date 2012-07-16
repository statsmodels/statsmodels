call tools\build_win_bdist64-py26.bat
call tools\build_win_bdist32-py26.bat
call tools\build_win_bdist64-py27.bat
call tools\build_win_bdist32-py27.bat
call tools\build_win_bdist32-py32.bat
call tools\build_win_bdist64-py32.bat
call python setup.py sdist --formats=zip,gztar
