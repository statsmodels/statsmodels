


filelist = ['AR_example.py', 'example_mle.py', 'example_wls.py', 'glm_example.py',
            'ols_fstat.py', 'Rpy_example.py']

for f in filelist:
    try:
        execfile(f)
    except:
        print "ERROR in example file", f

