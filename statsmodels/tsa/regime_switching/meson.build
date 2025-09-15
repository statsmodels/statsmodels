_hamilton_filter_pyx = custom_target(
  '_hamilton_filter_pyx',
  input: '_hamilton_filter.pyx.in',
  output: '_hamilton_filter.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module(
  '_hamilton_filter',
  [_hamilton_filter_pyx, math_pxd],
  include_directories: [inc_np, inc_sm],
  install: true
)

_kim_smoother_pyx = custom_target(
  '_kim_smoother_pyx',
  input: '_kim_smoother.pyx.in',
  output: '_kim_smoother.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module(
  '_kim_smoother',
  [_kim_smoother_pyx, math_pxd],
  include_directories: [inc_np, inc_sm],
  install: true
)
