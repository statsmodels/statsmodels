_arma_innovations_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_math_pxd, _cython_tree, _special_tree]
)

_arma_innovations_pyx = custom_target(
  '_arma_innovations_pyx',
  input: '_arma_innovations.pyx.in',
  output: '_arma_innovations.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_arma_innovations',
  _arma_innovations_gen.process(_arma_innovations_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/innovations',
)
