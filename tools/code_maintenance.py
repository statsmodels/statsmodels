"""
Code maintenance script modified from PyMC
"""

#!/usr/bin/env python
import sys
import os

# This is a function, not a test case, because it has to be run from inside
# the source tree to work well.

mod_strs = ['IPython', 'pylab', 'matplotlib', 'scipy','Pdb']

dep_files = {}
for mod_str in mod_strs:
    dep_files[mod_str] = []

def remove_whitespace(fname):
    # Remove trailing whitespace
    fd = open(fname,mode='U') # open in universal newline mode
    lines = []
    for line in fd.readlines():
        lines.append( line.rstrip() )
    fd.close()

    fd = open(fname,mode='w')
    fd.seek(0)
    for line in lines:
        fd.write(line+'\n')
    fd.close()
    # print 'Removed whitespace from %s'%fname

def find_whitespace(fname):
    fd = open(fname, mode='U')
    for line in fd.readlines():
        #print repr(line)
        if ' \n' in line:
            print fname
            break


# print
print_only = True

# ====================
# = Strip whitespace =
# ====================
for dirname, dirs, files in os.walk('.'):
    if dirname[1:].find('.')==-1:
        # print dirname
        for fname in files:
            if fname[-2:] in ['c', 'f'] or fname[-3:]=='.py' or fname[-4:] in ['.pyx', '.txt', '.tex', '.sty', '.cls'] or fname.find('.')==-1:
                # print fname
                if print_only:
                    find_whitespace(dirname + '/' + fname)
                else:
                    remove_whitespace(dirname + '/' + fname)


"""

# ==========================
# = Check for dependencies =
# ==========================
for dirname, dirs, files in os.walk('pymc'):
    for fname in files:
        if fname[-3:]=='.py' or fname[-4:]=='.pyx':
            if dirname.find('sandbox')==-1 and fname != 'test_dependencies.py'\
                and dirname.find('examples')==-1:
                for mod_str in mod_strs:
                    if file(dirname+'/'+fname).read().find(mod_str)>=0:
                        dep_files[mod_str].append(dirname+'/'+fname)


print 'Instances of optional dependencies found are:'
for mod_str in mod_strs:
    print '\t'+mod_str+':'
    for fname in dep_files[mod_str]:
        print '\t\t'+fname
if len(dep_files['Pdb'])>0:
    raise ValueError, 'Looks like Pdb was not commented out in '+', '.join(dep_files[mod_str])


"""
