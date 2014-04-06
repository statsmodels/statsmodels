"""run all examples to make sure we don't get an exception

Note:
If an example contaings plt.show(), then all plot windows have to be closed
manually, at least in my setup.

uncomment plt.show() to show all plot windows

"""
from __future__ import print_function
from statsmodels.compat import input
stop_on_error = True


filelist = ['example_glsar.py', 'example_wls.py', 'example_gls.py',
            'example_glm.py', 'example_ols_tftest.py',  # 'example_rpy.py',
            'example_ols.py', 'example_rlm.py',
            'example_discrete.py', 'example_predict.py',
            'example_ols_table.py',
            # time series
            'tsa/ex_arma2.py', 'tsa/ex_dates.py']

if __name__ == '__main__':
    #temporarily disable show
    import matplotlib.pyplot as plt
    plt_show = plt.show

    def noop(*args):
        pass

    plt.show = noop

    msg = """Are you sure you want to run all of the examples?
          This is done mainly to check that they are up to date.
          (y/n) >>> """

    cont = input(msg)
    if 'y' in cont.lower():
        for run_all_f in filelist:
            try:
                print('\n\nExecuting example file', run_all_f)
                print('-----------------------' + '-' * len(run_all_f))
                exec(open(run_all_f).read())
            except:
                # f might be overwritten in the executed file
                print('**********************' + '*' * len(run_all_f))
                print('ERROR in example file', run_all_f)
                print('**********************' + '*' * len(run_all_f))
                if stop_on_error:
                    raise

    # reenable show after closing windows
    plt.close('all')
    plt.show = plt_show
    plt.show()
