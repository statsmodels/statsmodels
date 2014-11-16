"""
Contains values used to approximate the critical value and p-value from DFGLS
statistics

These have been computed using the methodology of MacKinnon (1994) and (2010)
simulation. See dfgls_critival_values_simulation for implementation.
"""

from numpy import array

dfgls_cv_approx = {'c': array([[-2.56781793e+00, -2.05575392e+01, 1.82727674e+02,
                                -1.77866664e+03],
                               [-1.94363325e+00, -2.17272746e+01, 2.60815068e+02,
                                -2.26914916e+03],
                               [-1.61998241e+00, -2.32734708e+01, 3.06474378e+02,
                                -2.57483557e+03]]),
                   'ct': array([[-3.40689134, -21.69971242, 27.26295939, -816.84404772],
                                [-2.84677178, -19.69109364, 84.7664136, -799.40722401],
                                [-2.55890707, -19.42621991, 116.53759752, -840.31342847]])}

dfgls_tau_max = {'c': 13.365361509140614,
                 'ct': 8.73743383728356}

dfgls_tau_min = {'c': -17.561302895074206,
                 'ct': -13.681153542634465}

dfgls_tau_star = {'c': -0.4795076091714674,
                  'ct': -2.1960404365401298}

dfgls_large_p = {'c': array([0.50612497, 0.98305664, -0.05648525, 0.00140875]),
                 'ct': array([2.60561421, 1.67850224, 0.0373599, -0.01017936])}

dfgls_small_p = {'c': array([0.67422739, 1.25475826, 0.03572509]),
                 'ct': array([2.38767685, 1.57454737, 0.05754439])}