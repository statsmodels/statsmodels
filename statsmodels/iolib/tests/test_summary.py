from statsmodels.iolib.summary import (Summary, summary_params, 
        summary_model, summary_col)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import OrderedDict
import numpy as np

# Setup
data = sm.datasets.spector.load()
data.exog = sm.add_constant(data.exog, prepend=False)
mod = sm.OLS(data.endog, data.exog)
res = mod.fit()
mod_info = summary_model(res)
mod_para = summary_params(res)

def test_summary_params():
    output = np.array(summary_params(res, alpha=0.1))
    desire = np.array([[ 0.46385168,  0.16195635,  2.86405365,  0.00784052,  0.18834272,
         0.73936064],
       [ 0.01049512,  0.01948285,  0.53868506,  0.59436148, -0.02264776,
         0.04363801],
       [ 0.37855479,  0.13917274,  2.72003545,  0.01108768,  0.14180373,
         0.61530584],
       [-1.49801712,  0.52388862, -2.85941908,  0.00792932, -2.38922026,
        -0.60681398]])
    np.testing.assert_array_almost_equal(output, desire, 4)

def test_model_info():
    output = summary_model(res)
    desire = OrderedDict([('Model:', 'OLS'), ('Dependent Variable:', 'y'), ('No. Observations:', '    32'), ('Df Model:', '     3'), ('Df Residuals:', '    28'), ('R-squared:', '   0.416'), ('Adj. R-squared:', '   0.353'), ('AIC:', ' 33.9565'), ('BIC:', ' 39.8194'), ('Log-Likelihood:', ' -12.978'), ('Scale:', ' 0.15059')])
    np.testing.assert_(output == desire)

def test_add_df():
    smry = Summary()
    smry.add_df(mod_para, header=True, index=False, float_format="%.4g",
            align='l')
    smry.add_df(mod_para, header=True, index=True, float_format="%.4g",
            align='c')
    smry.add_df(mod_para, header=False, index=False, float_format="%.2f",
            align='r')
    output = smry.as_text()

    desire = '''
======================================================
 Coef.   Std.Err.    t      P>|t|     [0.025    0.975]
------------------------------------------------------
0.4639   0.162     2.864   0.007841  0.1321    0.7956 
0.0105   0.01948   0.5387  0.5944    -0.02941  0.0504 
0.3786   0.1392    2.72    0.01109   0.09347   0.6636 
-1.498   0.5239    -2.859  0.007929  -2.571    -0.4249
------------------------------------------------------
      Coef.  Std.Err.   t     P>|t|    [0.025   0.975]
------------------------------------------------------
x1    0.4639  0.162   2.864  0.007841  0.1321   0.7956
x2    0.0105 0.01948  0.5387  0.5944  -0.02941  0.0504
x3    0.3786  0.1392   2.72  0.01109  0.09347   0.6636
const -1.498  0.5239  -2.859 0.007929  -2.571  -0.4249
------------------------------------------------------
 0.46      0.16      2.86     0.01      0.13      0.80
 0.01      0.02      0.54     0.59     -0.03      0.05
 0.38      0.14      2.72     0.01      0.09      0.66
-1.50      0.52     -2.86     0.01     -2.57     -0.42
======================================================
'''
    np.testing.assert_(output == desire)

def test_add_dict():
    smry = Summary()
    smry.add_dict(mod_info, 1, 'l')
    smry.add_dict(mod_info, 2, 'c')
    smry.add_dict(mod_info, 3, 'r')
    output = smry.as_text()
    desire = '''
=======================================================================
Model:                                                          OLS    
Dependent Variable:                                             y      
No. Observations:                                               32     
Df Model:                                                       3      
Df Residuals:                                                   28     
R-squared:                                                      0.416  
Adj. R-squared:                                                 0.353  
AIC:                                                            33.9565
BIC:                                                            39.8194
Log-Likelihood:                                                 -12.978
Scale:                                                          0.15059
-----------------------------------------------------------------------
      Model:                 OLS         Adj. R-squared:         0.353 
Dependent Variable:           y                AIC:             33.9565
 No. Observations:            32               BIC:             39.8194
     Df Model:                3          Log-Likelihood:        -12.978
   Df Residuals:              28              Scale:            0.15059
    R-squared:              0.416                                      
-----------------------------------------------------------------------
             Model: OLS   Df Residuals:      28            BIC: 39.8194
Dependent Variable:   y      R-squared:   0.416 Log-Likelihood: -12.978
  No. Observations:  32 Adj. R-squared:   0.353          Scale: 0.15059
          Df Model:   3            AIC: 33.9565                        
=======================================================================
'''
    output == desire
    np.testing.assert_(output == desire)

def test_add_array():
    np.random.seed(1024)
    array2d = np.random.random((2,2))
    array3d = np.array([
    ['Row 1', .123456, 'Other text here'],
    ['Row 2', 'Some text over here', .654321],
    ['Row 3', 'Some text over here', 654321]
    ])
    smry = Summary()
    smry.add_array(array2d, float_format="%.4f", align='l')
    smry.add_array(array2d, float_format="%.2f", align='c')
    smry.add_array(array3d, align='c')
    smry.add_array(array3d, align='r')
    output = smry.as_text()
    desire = '''
=========================================
0.6477                             0.9969
0.5188                             0.6581
-----------------------------------------
0.65                                 1.00
0.52                                 0.66
-----------------------------------------
Row 1       0.123456      Other text here
Row 2 Some text over here     0.654321   
Row 3 Some text over here      654321    
-----------------------------------------
Row 1            0.123456 Other text here
Row 2 Some text over here        0.654321
Row 3 Some text over here          654321
=========================================
'''
    np.testing.assert_(output == desire)

def test_add_text():
    smry = Summary()
    array3d = np.array([
        ['Row 1', .123456, 'Other text here'],
        ['Row 2', 'Some text over here', .654321],
        ['Row 3', 'Some text over here', 654321]
        ])
    smry.add_array(array3d)
    smry.add_text('+ Boudin ribeye ham hock rump turducken, pig cow pork loin leberkas t-bone sausage strip steak. Ground round venison ham hock sausage bresaola capicola prosciutto shoulder swine. Spare ribs beef kielbasa salami fatback.')
    smry.add_text('+ Andouille short ribs doner corned beef ground round pig pork chop. Tail fatback biltong turkey jowl tri-tip venison spare ribs pancetta cow ham rump drumstick brisket corned beef.')
    output = smry.as_text()
    desire = '''
=========================================
Row 1            0.123456 Other text here
Row 2 Some text over here        0.654321
Row 3 Some text over here          654321
=========================================
+ Boudin ribeye ham hock rump turducken,
pig cow pork loin leberkas t-bone sausage
strip steak. Ground round venison ham
hock sausage bresaola capicola prosciutto
shoulder swine. Spare ribs beef kielbasa
salami fatback.
+ Andouille short ribs doner corned beef
ground round pig pork chop. Tail fatback
biltong turkey jowl tri-tip venison spare
ribs pancetta cow ham rump drumstick
brisket corned beef.'''
    np.testing.assert_(output == desire)

def test_as_latex():
    output = res.summary().as_latex()
    desire = '\\begin{table}\n\\caption{Results: Ordinary least squares} \\\\\n\\begin{tabular}{llll}\n\\hline\nModel:              & OLS   & Adj. R-squared: & 0.353    \\\\\nDependent Variable: & y     & AIC:            & 33.9565  \\\\\nNo. Observations:   & 32    & BIC:            & 39.8194  \\\\\nDf Model:           & 3     & Log-Likelihood: & -12.978  \\\\\nDf Residuals:       & 28    & Scale:          & 0.15059  \\\\\nR-squared:          & 0.416 &                 &          \\\\\n\\hline\n\\end{tabular}\n\\hline\n\\begin{tabular}{lrrrrrr}\n\\hline\n      &  Coef.  & Std.Err. &    t    & P>|t|  &  [0.025 &  0.975]  \\\\\n\\hline\nx1    &  0.4639 &   0.1620 &  2.8641 & 0.0078 &  0.1321 &  0.7956  \\\\\nx2    &  0.0105 &   0.0195 &  0.5387 & 0.5944 & -0.0294 &  0.0504  \\\\\nx3    &  0.3786 &   0.1392 &  2.7200 & 0.0111 &  0.0935 &  0.6636  \\\\\nconst & -1.4980 &   0.5239 & -2.8594 & 0.0079 & -2.5712 & -0.4249  \\\\\n\\hline\n\\end{tabular}\n\\hline\n\\begin{tabular}{llll}\n\\hline\nOmnibus:       & 0.176 & Durbin-Watson:    & 2.346  \\\\\nProb(Omnibus): & 0.916 & Jarque-Bera (JB): & 0.167  \\\\\nSkew:          & 0.141 & Prob(JB):         & 0.920  \\\\\nKurtosis:      & 2.786 & Condition No.:    & 176    \\\\\n\\hline\n\\end{tabular}\n\\end{table}'
    output == desire
    np.testing.assert_(output == desire)

def test_add_title():
    smry = Summary()
    smry.add_title('Test title')
    np.random.seed(1024)
    smry.add_array(np.random.random((2,2)))
    output = smry.as_text()
    desire = ''' Test title
=============
0.6477 0.9969
0.5188 0.6581
=============
'''
    output == desire
    np.testing.assert_(output == desire)

def test_add_base():
    smry = Summary()
    smry.add_base(res)
    output = smry.as_text()
    desire = '''           Results: Ordinary least squares
=====================================================
Model:                OLS    Adj. R-squared:  0.353  
Dependent Variable:   y      AIC:             33.9565
No. Observations:     32     BIC:             39.8194
Df Model:             3      Log-Likelihood:  -12.978
Df Residuals:         28     Scale:           0.15059
R-squared:            0.416                          
-----------------------------------------------------
       Coef.  Std.Err.    t    P>|t|   [0.025  0.975]
-----------------------------------------------------
x1     0.4639   0.1620  2.8641 0.0078  0.1321  0.7956
x2     0.0105   0.0195  0.5387 0.5944 -0.0294  0.0504
x3     0.3786   0.1392  2.7200 0.0111  0.0935  0.6636
const -1.4980   0.5239 -2.8594 0.0079 -2.5712 -0.4249
=====================================================
'''
    output == desire
    np.testing.assert_(output == desire)



# Setup summary_col
df = sm.datasets.spector.load_pandas().data
res1 = smf.ols('GPA ~ TUCE + PSI', df).fit()
res2 = smf.ols('GPA ~ TUCE + PSI + GRADE', df).fit()
res3 = smf.ols('GRADE ~ TUCE + PSI', df).fit()
results = [res1, res2, res3]

def test_col_default():
    output = summary_col(results).as_text()
    desire = '''
======================================
             GPA       GPA     GRADE  
--------------------------------------
GRADE     0.4885*** 0.4885***         
          (0.1706)  (0.1706)          
Intercept 2.3575*** 2.3575*** -0.5230 
          (0.4182)  (0.4182)  (0.4449)
PSI       -0.1878   -0.1878   0.3768**
          (0.1566)  (0.1566)  (0.1555)
TUCE      0.0307    0.0307    0.0320  
          (0.0192)  (0.0192)  (0.0201)
R2        0.342     0.342     0.245   
AIC       35.612    35.612    40.178  
N         32        32        32      
======================================
Standard errors in parentheses.
* p<.1, ** p<.05, ***p<.01'''
    np.testing.assert_(output == desire)

def test_col_starless():
    output = summary_col(results, stars=False).as_text()
    desire = '''
====================================
            GPA      GPA     GRADE  
------------------------------------
GRADE     0.4885   0.4885           
          (0.1706) (0.1706)         
Intercept 2.3575   2.3575   -0.5230 
          (0.4182) (0.4182) (0.4449)
PSI       -0.1878  -0.1878  0.3768  
          (0.1566) (0.1566) (0.1555)
TUCE      0.0307   0.0307   0.0320  
          (0.0192) (0.0192) (0.0201)
R2        0.342    0.342    0.245   
AIC       35.612   35.612   40.178  
N         32       32       32      
====================================
Standard errors in parentheses.'''
    np.testing.assert_(output == desire)

def test_col_float_format():
    output = summary_col(results, float_format="%.2f").as_text()
    desire = '''
================================
            GPA     GPA   GRADE 
--------------------------------
GRADE     0.49*** 0.49***       
          (0.17)  (0.17)        
Intercept 2.36*** 2.36*** -0.52 
          (0.42)  (0.42)  (0.44)
PSI       -0.19   -0.19   0.38**
          (0.16)  (0.16)  (0.16)
TUCE      0.03    0.03    0.03  
          (0.02)  (0.02)  (0.02)
R2        0.342   0.342   0.245 
AIC       35.612  35.612  40.178
N         32      32      32    
================================
Standard errors in parentheses.
* p<.1, ** p<.05, ***p<.01'''
    np.testing.assert_(output == desire)


def test_col_model_names():
    output = summary_col(results, model_names=['andouillette et boudin','b','c']).as_text()
    desire = '''
===================================================
          andouillette et boudin     b        c    
---------------------------------------------------
GRADE                            0.4885***         
                                 (0.1706)          
Intercept 2.1021***              2.3575*** -0.5230 
          (0.4566)               (0.4182)  (0.4449)
PSI       -0.0037                -0.1878   0.3768**
          (0.1596)               (0.1566)  (0.1555)
TUCE      0.0463**               0.0307    0.0320  
          (0.0206)               (0.0192)  (0.0201)
R2        0.150                  0.342     0.245   
AIC       41.833                 35.612    40.178  
N         32                     32        32      
===================================================
Standard errors in parentheses.
* p<.1, ** p<.05, ***p<.01'''
    np.testing.assert_(output == desire)

def test_col_dict_info():
    custom_info = {'N': lambda x: str(int(x.nobs)), 
                   'BIC': lambda x: '%.3f' % x.aic, 
                   'R2-adj': lambda x: '%.3f' % x.rsquared_adj, 
                   'F': lambda x: '%.3f' % x.fvalue}
    output = summary_col(results, info_dict=custom_info).as_text()
    desire = '''
======================================
             GPA       GPA     GRADE  
--------------------------------------
GRADE     0.4885*** 0.4885***         
          (0.1706)  (0.1706)          
Intercept 2.3575*** 2.3575*** -0.5230 
          (0.4182)  (0.4182)  (0.4449)
PSI       -0.1878   -0.1878   0.3768**
          (0.1566)  (0.1566)  (0.1555)
TUCE      0.0307    0.0307    0.0320  
          (0.0192)  (0.0192)  (0.0201)
R2-adj    0.272     0.272     0.193   
F         4.860     4.860     4.700   
BIC       35.612    35.612    40.178  
N         32        32        32      
======================================
Standard errors in parentheses.
* p<.1, ** p<.05, ***p<.01'''
    np.testing.assert_(output == desire)
