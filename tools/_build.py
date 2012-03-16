"""
The cython function was adapted from scikits-image (http://scikits-image.org/)
"""

import sys
import os
import shutil
import subprocess
import platform
from distutils.dist import Distribution
from distutils.command.config import config as distutils_config
from distutils import log
import optparse # deprecated in 2.7 for argparse

dummy_c_text = r'''
/* This file is generated from statsmodels/tools/_build.py to */
void do_nothing(void);
int main(void) {
    do_nothing();
    return 0;
}
'''


def has_c_compiler():
    c = distutils_config(Distribution())
    if platform.system() == "Windows": # HACK
        # this doesn't matter because mingw won't be given at install step
        # check if mingw was given in compiler options
        #parser = optparse.OptionParser()
        #parser.add_option('-c', '--compiler', dest='compiler')
        #options, args = parser.parse_args()
        #if options.compiler and 'mingw' in options.compiler:
        #    return True

        # if not, then check to see if compiler is set in disutils.cfg
        try: # Josef's code to check the distutils.cfg file
            c.distribution.parse_config_files(c.distribution.find_config_files())
            # this will raise a key error if there's not one
            c.distribution.command_options['build']['compiler'][1]
            return True
        except:
            pass
    # just see if there's a system compiler
    try:
        success = c.try_compile(dummy_c_text)
        return True
    except:
        log.info("No C compiler detected. Not installing Cython version "
                 "of files.")
        return False


def cython(pyx_files, working_path=''):
    """Use Cython to convert the given files to C.

    Parameters
    ----------
    pyx_files : list of str
        The input .pyx files.

    """
    # Do not build cython files if target is clean
    if len(sys.argv) >= 2 and sys.argv[1] == 'clean':
        return

    try:
        import Cython
    except ImportError:
        # If cython is not found, we do nothing -- the build will make use of
        # the distributed .c files
        print("Cython not found; falling back to pre-built %s" \
              % " ".join([f.replace('.pyx', '.c') for f in pyx_files]))
    else:
        for pyxfile in [os.path.join(working_path, f) for f in pyx_files]:

            #TODO: replace this with already written hash_funcs once merged
            # if the .pyx file stayed the same, we don't need to recompile
            #if not _changed(pyxfile):
            #    continue

            c_file = pyxfile[:-4] + '.c'

            # run cython compiler
            cmd = 'cython -o %s %s' % (c_file, pyxfile)
            print(cmd)

            if platform.system() == 'Windows':
                status = subprocess.call(
                    [sys.executable,
                     os.path.join(os.path.dirname(sys.executable),
                                  'Scripts', 'cython.py'),
                     '-o', c_file, pyxfile],
                    shell=True)
            else:
                status = subprocess.call(['cython', '-o', c_file, pyxfile])
