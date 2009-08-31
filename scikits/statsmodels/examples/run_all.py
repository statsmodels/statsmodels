


filelist = ['example_glsar.py', 'example_wls.py',
            'example_glm.py', 'example_ols_tftest.py', 'example_rpy.py',
            'example_gls.py', 'example_ols.py', 'example_rlm.py']

for f in filelist:
    try:
        execfile(f)
    except:
        print "*********************"
        print "ERROR in example file", f
        print "*********************"

