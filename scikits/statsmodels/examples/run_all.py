


filelist = ['example_glsar.py', 'example_mle.py', 'example_wls.py',
            'example_glm.py', 'example_ols_tftest.py', 'example_rpy.py']

for f in filelist:
    try:
        execfile(f)
    except:
        print "*********************"
        print "ERROR in example file", f
        print "*********************"

