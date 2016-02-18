"""
A collection of utilities to see if new ReST files need to be automatically
generated from certain files in the project (examples, datasets).
"""
import os
from statsmodels.compat import cPickle

file_path = os.path.dirname(__file__)

def get_hash(f):
    """
    Gets hexadmecimal md5 hash of a string
    """
    import hashlib
    m = hashlib.md5()
    m.update(f)
    return m.hexdigest()

def update_hash_dict(filehash, filename):
    """
    Opens the pickled hash dictionary, adds an entry, and dumps it back.
    """
    try:
        with open(file_path + '/hash_dict.pickle', 'rb') as f:
            hash_dict = cPickle.load(f)
    except IOError:
        hash_dict = {}
    hash_dict.update({filename: filehash})
    with open(os.path.join(file_path, 'hash_dict.pickle'), 'wb') as f:
        cPickle.dump(hash_dict, f)

def check_hash(rawfile, filename):
    """
    Returns True if hash does not match the previous one.
    """
    try:
        with open(file_path + '/hash_dict.pickle', 'rb') as f:
            hash_dict = cPickle.load(f)
    except IOError:
        hash_dict = {}
    try:
        checkhash = hash_dict[filename]
    except:
        checkhash = None

    filehash = get_hash(rawfile)
    if filehash == checkhash:
        return False, None
    return True, filehash
