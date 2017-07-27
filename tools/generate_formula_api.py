#!/usr/bin/env python
"""
This will generate an API file for formula. in dir/statsmodels/formula/api.py

It first builds statsmodels in place, then generates the file. It's to be run
by developers to add files to the formula API without having to maintain this
by hand.

usage

generate_formula_api /home/skipper/statsmodels/statsmodels/
"""

import sys
import os

def iter_subclasses(cls, _seen=None, template_classes=[]):
    """
    Generator to iterate over all the subclasses of Model. Based on

    http://code.activestate.com/recipes/576949-find-all-subclasses-of-a-given-class/

    Yields class
    """
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None:
        _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen and sub.__name__ not in template_classes:
            _seen.add(sub)
            # we don't want to yield the templates, but we do want to
            # recurse on them
            yield sub
        for sub in iter_subclasses(sub, _seen, template_classes):
            yield sub

def write_formula_api(directory):
    template_classes = ['DiscreteModel', 'BinaryModel', 'MultinomialModel',
                        'OrderedModel', 'CountModel',
                      'LikelihoodModel', 'GenericLikelihoodModel',
                      'TimeSeriesModel',
                      # this class should really be deleted
                      'ARIMAProcess',
                      # these need some more work, so don't expose them
                      'ARIMA', 'VAR', 'SVAR', 'AR', 'NBin', 'NbReg', 'ARMA',
                      ]

    fout = open(os.path.join(directory, 'statsmodels', 'formula', 'api.py'),
                                        'w')
    for model in iter_subclasses(Model, template_classes=template_classes):
        print "Generating API for %s" % model.__name__
        fout.write(
                'from '+model.__module__+' import ' + model.__name__ + '\n'
                )
        fout.write(
                model.__name__.lower() +' = '+ model.__name__ +'.from_formula\n'
                )
    fout.close()

if __name__ == "__main__":
    import statsmodels.api as sm
    print "Generating formula API for statsmodels version %s" % sm.version.full_version
    directory = sys.argv[1]
    cur_dir = os.path.dirname(__file__)
    os.chdir(directory)
    # it needs to be installed to walk the whole subclass chain?
    from statsmodels.base.model import Model
    write_formula_api(directory)
