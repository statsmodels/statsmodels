'''
Module for writing a test file from NIST data

The following reads model name, its function expression
and jacobian from a configuration file, 'model_list.cfg'
and creates a test module from its corresponding NIST
data file.
It is advised that no changes are done in output file.
Any desired modification should be generalised and added
in this module

'''


__PROLOGUE__1 = '''
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS

'''
__PROLOGUE__2 = '''
import numpy as np
import statsmodels.api as sm
from numpy.testing import assert_almost_equal, assert_
'''

def create_Modelclass(cfgfile):
    with open(str(cfgfile)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = line.strip().split(':')
            model_name = line[0].strip()
            model_func = line[1].strip()
            model_jacob = line[2].strip()
            NIST_results = read_NISTdata(model_name)
            s = __PROLOGUE__1
            s += Create_Nonlinls_class(model_name, 
                 model_func, model_jacob, NIST_results)
            s += Create_test_class(model_name,NIST_results)
            filename = model_name+'_testclass.py'            
            with open(filename,'w') as fname:
                fname.write(s)            
            s = __PROLOGUE__2
            s += 'from '+model_name+'_testclass import TestNonlinearLS\n'
            s += Create_Nonlinls_test(model_name)
            filename = 'test_'+model_name+'.py'
            with open(filename,'w') as fname:
                fname.write(s)

def Create_Nonlinls_class(model_name, model_func, model_jacob,
                          NIST_results):
    s = 'class func'+model_name+'(NonlinearLS):\n'
    s += '\n    def expr(self, params, exog=None):\n'
    s += '        if exog is None:\n'
    s += '            x = self.exog\n'
    s += '        else:\n'
    s += '            x = exog\n'
    s += '        '
    nparams = int(NIST_results['nparams'])
    for i in range(nparams):
        if i == 0:
            s += 'b'+str(i+1)
        elif i > 0:
            s += ', '+'b'+str(i+1)
    s += ' = params\n'
    s += '        return '+model_func+'\n'
    s += '\nclass func'+model_name+'_J(NonlinearLS):\n'
    s += '\n    def expr(self, params, exog=None):\n'
    s += '        if exog is None:\n'
    s += '            x = self.exog\n'
    s += '        else:\n'
    s += '            x = exog\n'
    s += '        '
    nparams = int(NIST_results['nparams'])
    for i in range(nparams):
        if i == 0:
            s += 'b'+str(i+1)
        elif i > 0:
            s += ', '+'b'+str(i+1)
    s += ' = params\n'
    s += '        return '+model_func+'\n'
    s += '\n    def jacobian(self, params, exog=None):\n'
    s += '        if exog is None:\n'
    s += '            x = self.exog\n'
    s += '        else:\n'
    s += '            x = exog\n'
    s += '        '
    nparams = int(NIST_results['nparams'])
    for i in range(nparams):
        if i == 0:
            s += 'b'+str(i+1)
        elif i > 0:
            s += ', '+'b'+str(i+1)
    s += ' = params\n'
    s += '        return '+model_jacob+'\n\n'
    return s

def Create_test_class(model_name,NIST_results):
    s = 'class TestNonlinearLS(object):'
    s += '\n    def setup(self):\n'
    x = str('        x = np.array('+str(NIST_results['x'])+
        ')\n').replace(',',str(',\n'+' '*18))
    y = str('        y = np.array('+str(NIST_results['y'])+
        ')\n').replace(',',str(',\n'+' '*18))
    s += x + y
    for k in NIST_results.keys():
        if k in list(['x','y']):
            continue
        else:
           s += '        self.'+str(k)+'='+str(NIST_results[k])+'\n'
    s += '\n'
    s += '        mod1 = func'+model_name+'(y, x)\n'
    s += '        self.res_start1 = mod1.fit(self.start_value1)\n'
    s += '        mod2 = func'+model_name+'(y, x)\n'
    s += '        self.res_start2 = mod2.fit(self.start_value2)\n'
    s += '        mod1_J = func'+model_name+'_J(y, x)\n'
    s += '        self.resJ_start1 = mod1_J.fit(self.start_value1)\n'
    s += '        mod2_J = func'+model_name+'_J(y, x)\n'
    s += '        self.resJ_start2 = mod2_J.fit(self.start_value2)\n'
    return s

def Create_Nonlinls_test(model_name):
    s = '\nclass Test'+model_name+'(TestNonlinearLS):\n'
    s += '\n    def test_basic(self):\n'
    s += '        res1 = self.res_start1\n'
    s += '        res2 = self.res_start2\n'
    s += '        res1J = self.resJ_start1\n'
    s += '        res2J = self.resJ_start2\n'
    s += '        certified = self.Cert_parameters\n'    
    s += '        assert_almost_equal(res1.params,certified)\n'
    s += '        assert_almost_equal(res2.params,certified)\n'
    s += '        assert_almost_equal(res1J.params,certified)\n'
    s += '        assert_almost_equal(res2J.params,certified)\n'
    return s

def read_NISTdata(Model):
    with open (str(Model)+'.dat') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Model:'):
                i = lines.index(line)
                nparams=lines[i+1].strip()[0]
                break
        Certified_Val = lines[40:58]
        Data = lines[60:]

        start1, start2, params, stddev = [], [], [], []
        for x in range(len(Certified_Val)):
            Certified_Val[x] = Certified_Val[x].strip()
            contents = Certified_Val[x].split(':')
            if Certified_Val[x].startswith('b'):
                all_values = Certified_Val[x].split()
                start1.append(float(all_values[2]))
                start2.append(float(all_values[3]))
                params.append(float(all_values[4]))
                stddev.append(float(all_values[5]))
            elif Certified_Val[x].startswith('Residual Sum of Squares'):
                res_squares = float(contents[1])
            elif Certified_Val[x].startswith('Residual Standard Deviation'):
                res_stddev = float(contents[1])
            elif Certified_Val[x].startswith('Degrees of Freedom'):
                deg_free = int(contents[1])
            elif Certified_Val[x].startswith('Number of Observations'):
                nobs = int(contents[1])
            else:
                continue

        y, x = [], []
        for k in range(len(Data)):        
            i,j = Data[k].strip().split()
            y.append(float(i))
            x.append(float(j))

    return {'y': y, 'x': x, 'start_value1':start1, 'start_value2':start2,
            'Cert_parameters':params, 'Cert_stddev':stddev, 
            'Res_sum_squares':res_squares,'Res_stddev':res_stddev,
            'Degrees_free':deg_free, 'Nobs_data':nobs, 'nparams':nparams }

if __name__ == '__main__':
    a = create_Modelclass('model_list.cfg')

