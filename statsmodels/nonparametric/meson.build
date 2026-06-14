py.extension_module(
    '_smoothers_lowess',
    '_smoothers_lowess.pyx',
    install: true,
    include_directories: [inc_np],
    subdir: 'statsmodels/nonparametric',
    c_args: cython_c_args,
)

py.extension_module(
    'linbin',
    'linbin.pyx',
    install: true,
    include_directories: [inc_np],
    subdir: 'statsmodels/nonparametric',
    c_args: cython_c_args,
)
