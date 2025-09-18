rs_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_math_pxd, _cython_tree, _special_tree]
)


_hamilton_filter_pyx = custom_target(
  '_hamilton_filter_pyx',
  input: '_hamilton_filter.pyx.in',
  output: '_hamilton_filter.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_hamilton_filter',
  rs_cython_gen.process(_hamilton_filter_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/regime_switching',
)


_kim_smoother_pyx = custom_target(
  '_kim_smoother_pyx',
  input: '_kim_smoother.pyx.in',
  output: '_kim_smoother.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_kim_smoother',
  rs_cython_gen.process(_kim_smoother_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/regime_switching',
)
