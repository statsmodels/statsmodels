'''Save a set of numpy arrays to a python module file that can be imported

Author : Josef Perktold
'''
from __future__ import print_function
from statsmodels.compat.python import iterkeys
import numpy as np

class HoldIt(object):
    '''Class to write numpy arrays into a python module

    Calling save on the instance of this class write all attributes of the
    instance into a module file. For details see the save method.

    '''

    def __init__(self, name):
        self.name = name
    def save(self, what=None, filename=None, header=True, useinstance=True,
             comment=None, print_options=None):
        '''write attributes of this instance to python module given by filename

        Parameters
        ----------
        what : list or None
            list of attributes that are added to the module. If None (default)
            then all attributes in __dict__ that do not start with an underline
            will be saved.
        filename : string
            specifies filename with path. If the file does not exist, it will be
            created. If the file is already exists, then the new data will be
            appended to the file.
        header : bool
            If true, then the imports of the module and the class definition are
            written before writing the data.
        useinstance : bool
            If true, then the data in the module are attached to an instance of a
            holder class. If false, then each array will be saved as separate
            variable.
        comment : string
            If comment is not empty then this string will be attached as a
            description comment to the data instance in the saved module.
        print_options : dict or None
            The print_options for the numpy arrays will be updated with this.
            see notes

        Notes
        -----
        The content of an numpy array are written using repr, which can be
        controlled with the np.set_printoptions. The numpy default is updated
        with:  precision=20, linewidth=100, nanstr='nan', infstr='inf'

        This should provide enough precision for double floating point numbers.
        If one array has more than 1000 elements, then threshold should be
        overwritten by the user, see keyword argument print_options.
        '''

        print_opt_old = np.get_printoptions()
        print_opt = dict(precision=20, linewidth=100, nanstr='nan',
                            infstr='inf')
        if print_options:
            print_opt.update(print_options)
        np.set_printoptions(**print_opt)
        #precision corrects for non-scientific notation
        if what is None:
            what = (i for i in self.__dict__ if i[0] != '_')
        if header:
            txt = ['import numpy as np\n'
                   'from numpy import array, rec, inf, nan\n\n']
            if useinstance:
                txt.append('class Holder(object):\n    pass\n\n')
        else:
            txt = []

        if useinstance:
            txt.append('%s = Holder()' % self.name)
            prefix = '%s.' % self.name
        else:
            prefix = ''

        if not comment is None:
            txt.append("%scomment = '%s'" % (prefix, comment))

        for x in what:
            txt.append('%s%s = %s' % (prefix, x, repr(getattr(self,x))))
        txt.extend(['','']) #add empty lines at end
        if not filename is None:
            file(filename, 'a+').write('\n'.join(txt))
        np.set_printoptions(**print_opt_old)
        self._filename = filename
        self._useinstance = useinstance
        self._what = what
        return txt

    def verify(self):
        '''load the saved module and verify the data

        This tries several ways of comparing the saved and the attached data,
        but might not work for all possible data structures.

        Returns
        -------
        all_correct : bool
            true if no differences are found, for floating point numbers
            rtol=1e-16, atol=1e-16 is used to determine equality (allclose)
        correctli : list
            list of attribute names that compare as equal
        incorrectli : list
            list of attribute names that did not compare as equal, either
            because they differ or because the comparison does not handle the
            data structure correctly

        '''
        module = __import__(self._filename.replace('.py',''))
        if not self._useinstance:
            raise NotImplementedError('currently only implemented when'
                                      'useinstance is true')
        data = getattr(module, self.name)
        correctli = []
        incorrectli = []

        for d in self._what:
            self_item = getattr(data, d)
            saved_item = getattr(data, d)
            #print(d)
            #try simple equality
            correct = np.all(self.item == saved_item)
            #try allclose
            if not correct and not self.item.dtype == np.dtype('object'):
                correct = np.allclose(self_item, saved_item,
                                  rtol=1e-16, atol=1e-16)
                if not correct:
                    import warnings
                    warnings.warn("inexact precision in "+d, RuntimeWarning)
            #try iterating, if object array
            if not correct:
                correlem =[np.all(data[d].item()[k] ==
                                  getattr(testsave.var_results, d).item()[k])
                           for k in iterkeys(data[d].item())]
                if not correlem:
                    #print(d, "wrong")
                    incorrectli.append(d)
            correctli.append(d)

        return len(incorrectli)==0, correctli, incorrectli



if __name__ == '__main__':
    data = np.load(r"E:\Josef\eclipsegworkspace\statsmodels-josef-experimental-030\dist\statsmodels-0.3.0dev_with_Winhelp_a2\statsmodels-0.3.0dev\scikits\statsmodels\tsa\vector_ar\tests\results\vars_results.npz")
    res_var =  HoldIt('var_results')
    for d in data:
        setattr(res_var, d, data[d])
    np.set_printoptions(precision=120, linewidth=100)
    res_var.save(filename='testsave.py', header=True,
                  comment='VAR test data converted from vars_results.npz')

    import testsave

    for d in data:
        print(d)
        correct = np.all(data[d] == getattr(testsave.var_results, d))
        if not correct and not data[d].dtype == np.dtype('object'):
            correct = np.allclose(data[d], getattr(testsave.var_results, d),
                              rtol=1e-16, atol=1e-16)
            if not correct: print("inexact precision")
        if not correct:
            correlem =[np.all(data[d].item()[k] ==
                              getattr(testsave.var_results, d).item()[k])
                       for k in iterkeys(data[d].item())]
            if not correlem:
                print(d, "wrong")

    print(res_var.verify())

