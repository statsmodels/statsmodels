#! /usr/bin/env python

import os
import sys
import re
import pickle
from StringIO import StringIO

# 3rd party
from matplotlib import pyplot as plt

# Ours
import hash_funcs

#----------------------------------------------------
# Globals
#----------------------------------------------------
# these files do not get made into .rst files because of
# some problems, they may need a simple cleaning up
exclude_list = ['example_rst.py','run_all.py', 'lagpolynomial.py',
                'ex_generic_mle_tdist.py', 'ex_generic_mle.py',
                'es_misc_poisson2.py', 'example_glsar.py',
                'ex_generic_mle_t.py', 'ex_misc_tmodel.py', 'try_polytrend.py',
                'example_rpy.py','t_est_rlm.py', 'ar1cholesky.py',
                'compare_arma.py', 'ex_arma.py', 'ex_coint.py', 'try_ar.py',
                'ex_var_reorder.py', 'arma_plots.py']

file_path = os.path.dirname(__file__)
docs_rst_dir = os.path.realpath(os.path.join(file_path,
                    '../scikits/statsmodels/docs/source/examples/generated/'))
example_dir = os.path.realpath(os.path.join(file_path,
                    '../scikits/statsmodels/examples/'))


def parse_docstring(block):
    """
    Strips the docstring from a string representation of the file.
    Returns the docstring and block without it
    """
    ds = "\"{3}|'{3}"
    try:
        start = re.search(ds, block).end()
        end = re.search(ds, block[start:]).start()
    except: #TODO: make more informative
        raise IOError("File does not have a docstring?")
    docstring = block[start:start+end]
    block = block[start+end+3:]
    return docstring.strip(), block

def parse_file(block):
    """
    Block is a raw string file.
    """
    docstring, block = parse_docstring(block)
    outfile = [docstring,'='*len(docstring),'']
    block = block.split('\n')

    # iterate through the rest of block, anything in comments is stripped of #
    # anything else is fair game to go in an ipython directive
    code_snippet = False
    for line in block:
        if not len(line):
            continue

        if line.startswith('#') and not (line.startswith('#%') or
                line.startswith('#@')):
            # on some ReST text
            if code_snippet: # were on a code snippet
                outfile.append('')
                code_snippet = False
            line = line.strip()
            outfile.append(line[1:].strip())
        else:
            if not code_snippet: # new code block
                outfile.append('\n.. ipython:: python\n')
                code_snippet = True
            # handle decorators and magic functions
            if line.startswith('#%') or line.startswith('#@'):
                line = line[1:]
            outfile.append('   '+line.strip('\n'))
    return '\n'.join(outfile)

def write_file(outfile, rst_file_pth):
    """
    Write outfile to rst_file_pth
    """
    print "Writing ", os.path.basename(rst_file_pth)
    write_file = open(rst_file_pth, 'w')
    write_file.writelines(outfile)
    write_file.close()

def restify(example_file):
    """
    Open the file
    Check the hash
    If needs updating, update the hash and
        Parse the file
        Write the new .rst
        Update the hash_dict
    """
    filename = os.path.basename(example_file)
    write_filename = os.path.join(docs_rst_dir,filename[:-2] + 'rst')
    example_file = open(example_file, 'r').read()
    #to_write, filehash = hash_funcs.check_hash(example_file, filename)
    to_write = True
    if to_write:
        rst_file = parse_file(example_file)
        write_file(rst_file, write_filename)
        #hash_funcs.update_hash_dict(filehash, filename)

if __name__ == "__main__":
    if len(sys.argv) > 1: # given a file,files to process, no help flag yet
        for example_file in sys.argv[1:]:
            restify(example_file)

    else: # process the whole directory
        for root, dirnames, filenames in os.walk(example_dir):
            for example in filenames:
                if (not example.endswith('.py') or example in exclude_list):
                    continue
                example_file = os.path.join(root,example)
                restify(example_file)
