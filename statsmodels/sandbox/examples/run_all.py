'''run all examples to make sure we don't get an exception

Note:
If an example contaings plt.show(), then all plot windows have to be closed
manually, at least in my setup.

uncomment plt.show() to show all plot windows

'''
from statsmodels.compat.python import input
stop_on_error = True


filelist = ['example_pca.py', 'example_sysreg.py', 'example_mle.py',
#            'example_gam.py', # exclude, currently we are not working on it
            'example_pca_regression.py']

cont = input("""Are you sure you want to run all of the examples?
This is done mainly to check that they are up to date.
(y/n) >>> """)
if 'y' in cont.lower():
    for run_all_f in filelist:
        try:
            print("Executing example file", run_all_f)
            print("-----------------------" + "-"*len(run_all_f))
            exec(open(run_all_f).read())
        except:
            #f might be overwritten in the executed file
            print("*********************")
            print("ERROR in example file", run_all_f)
            print("**********************" + "*"*len(run_all_f))
            if stop_on_error:
                raise
#plt.show()
#plt.close('all')
#close doesn't work because I never get here without closing plots manually
