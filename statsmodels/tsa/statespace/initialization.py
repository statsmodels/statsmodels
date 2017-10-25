#!/usr/bin/env python
# -*- coding: utf-8 -*-


def _initialize_error_cov(k_endog, error_cov_type):
    # equiv: scipy.misc.comb(k_endog, {'scalar': 0,
    #                                   'diagonal': 1,
    #                                   'unstructured': 2}[error_cov_type])
    if error_cov_type == 'scalar':
        nparams = 1
    elif error_cov_type == 'diagonal':
        nparams = k_endog
    elif error_cov_type == 'unstructured':
        nparams = int(k_endog * (k_endog + 1) / 2)
    return nparams
