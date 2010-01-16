

stop_on_error = True


filelist = ['example_glsar.py', 'example_wls.py', 'example_gls.py',
            'example_glm.py', 'example_ols_tftest.py', 'example_rpy.py',
            'example_ols.py', 'example_ols_minimal.py', 'example_rlm.py',
            'example_discrete.py']

cont = raw_input("""Are you sure you want to run all of the examples?
This is done mainly to check that they are up to date.
(y/n) >>> """)
if 'y' in cont.lower():
    for f in filelist:
        try:
            execfile(f)
        except:
            print "*********************"
            print "ERROR in example file", f
            print "*********************"
            if stop_on_error:
                raise

