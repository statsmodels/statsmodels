subdir('_filters')
subdir('_smoothers')

_statspece_tree = [
    fs.copyfile('__init__.py'),
    fs.copyfile('_cfa_simulation_smoother.pxd'),
    fs.copyfile('_initialization.pxd'),
    fs.copyfile('_kalman_filter.pxd'),
    fs.copyfile('_kalman_smoother.pxd'),
    fs.copyfile('_representation.pxd'),
    fs.copyfile('_simulation_smoother.pxd'),
    fs.copyfile('_tools.pxd')
]


statspece_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_math_pxd, _cython_tree, _special_tree, _statspece_filters_tree, _statspece_smoothers_tree, _statspece_tree]
)


_cfa_simulation_smoother_pyx = custom_target(
  '_cfa_simulation_smoother_pyx',
  input: '_cfa_simulation_smoother.pyx.in',
  output: '_cfa_simulation_smoother.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_cfa_simulation_smoother',
  statspece_cython_gen.process(_cfa_simulation_smoother_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  link_args: version_link_args,
  install: true,
  subdir: 'statsmodels/tsa/statespace',
)

_initialization_pyx = custom_target(
  '_initialization_pyx',
  input: '_initialization.pyx.in',
  output: '_initialization.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_initialization',
  statspece_cython_gen.process(_initialization_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  link_args: version_link_args,
  install: true,
  subdir: 'statsmodels/tsa/statespace',
)

_kalman_filter_pyx = custom_target(
  '_kalman_filter_pyx',
  input: '_kalman_filter.pyx.in',
  output: '_kalman_filter.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_kalman_filter',
  statspece_cython_gen.process(_kalman_filter_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  link_args: version_link_args,
  install: true,
  subdir: 'statsmodels/tsa/statespace',
)

_kalman_smoother_pyx = custom_target(
  '_kalman_smoother_pyx',
  input: '_kalman_smoother.pyx.in',
  output: '_kalman_smoother.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_kalman_smoother',
  statspece_cython_gen.process(_kalman_smoother_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  link_args: version_link_args,
  install: true,
  subdir: 'statsmodels/tsa/statespace',
)

_representation_pyx = custom_target(
  '_representation_pyx',
  input: '_representation.pyx.in',
  output: '_representation.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_representation',
  statspece_cython_gen.process(_representation_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  link_args: version_link_args,
  install: true,
  subdir: 'statsmodels/tsa/statespace',
)

_simulation_smoother_pyx = custom_target(
  '_simulation_smoother_pyx',
  input: '_simulation_smoother.pyx.in',
  output: '_simulation_smoother.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_simulation_smoother',
  statspece_cython_gen.process(_simulation_smoother_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  link_args: version_link_args,
  install: true,
  subdir: 'statsmodels/tsa/statespace',
)

_tools_pyx = custom_target(
  '_tools_pyx',
  input: '_tools.pyx.in',
  output: '_tools.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_tools',
  statspece_cython_gen.process(_tools_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  link_args: version_link_args,
  install: true,
  subdir: 'statsmodels/tsa/statespace',
)
