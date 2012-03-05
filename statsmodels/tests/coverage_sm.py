'''
create html coverage report using coverage

Note that this will work on the *installed* version of statsmodels; however,
the script should be run from the source tree's test directory.
'''

import sys
import statsmodels as sm
from coverage import coverage

# the generated html report will be placed in the tests directory
report_directory = 'coverage_report_html'

cov = coverage()
cov.start() # start logging coverage
sm.test()
cov.stop() # stop the logging coverage
cov.save() # save the logging coverage to ./.coverage
modpath = sm.__file__.strip('__init__.pyc') # get install directory
# set the module names to statsmodels.path.to.module
modnames = ['statsmodels.'+f.replace(modpath,'').replace('/',
        '.').replace('.py','') for f in cov.data.executed_files() if
        'statsmodels' in f]
# save only the use modules to the html report
cov.html_report([sys.modules[mn] for mn in modnames if mn in sys.modules],
                directory=report_directory)
