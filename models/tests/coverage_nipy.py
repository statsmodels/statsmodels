'''create html coverage report using coverage

works if nipy is on python path but not in pythons sitepackages
'''

import sys
import nipy.fixes.scipy.stats
from coverage import coverage
cov = coverage()
cov.start()
nipy.fixes.scipy.stats.test()
cov.stop()
cov.save()
#cov.html_report(directory='covhtml')
modpath = '/home/skipper/nipy/skipper-working/nipy/fixes/scipy/stats/'
modnames = [f.replace(modpath,'').replace('/','.').replace('.py','')
            for f in cov.data.executed_files() if 'nipy' in f]
cov.html_report([sys.modules[mn] for mn in modnames if mn in sys.modules],
                directory='html-skip')
