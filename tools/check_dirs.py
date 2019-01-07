'''helper script to check which directories are not on python path

all test folders should have and ``__init__.py``

'''

import os

root = '../statsmodels'

print('base, dnames, len(fnames), n_py')
for base, dnames, fnames in os.walk(root):
    if '__init__.py' not in fnames:
        # I have some empty directories when I switch git branches
        if (len(dnames) + len(fnames)) != 0:
            n_py = len([f for f in fnames if f[-3:] == '.py'])
            if n_py > 0:
                print(base, dnames, len(fnames), n_py)

    if '__pycache__' in dnames:
        dnames.remove('__pycache__')
    if 'src' in dnames:
        dnames.remove('src')
