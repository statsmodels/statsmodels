_arma_innovations_pyx = custom_target(
  '_arma_innovations_pyx',
  input: '_arma_innovations.pyx.in',
  output: '.',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module(
  '_arma_innovations',
  _arma_innovations_pyx,
  include_directories: [inc_np],
  install: true,
  subdir: 'statsmodels/tsa/innovations',
)
