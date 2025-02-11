from __future__ import print_function

import sys
if sys.version_info[0] == 2:
    _strobj = basestring
    _xrange = xrange
elif sys.version_info[0] == 3:
    _strobj = str
    _xrange = range

# stdlib
##import csv
from collections import OrderedDict
from copy import copy

# non stdlib
import numpy as np
import scipy
import scipy.stats
import scipy.signal
import pandas

# local
from ..texttable import _str
from ..texttable import Texttable as TextTable

def noncentrality_parameter(ss, sse, nobs):
    """
    Calculates non-centrality parameter for observed power estimates

    Assumes balanced design
    """
    
    p_eta2 = float(ss)/(float(ss) + float(sse))
    return (p_eta2/(1 - p_eta2))*nobs

def observed_power(df,dfe,nc,alpha=0.05,eps=1.0):
    """
    Power estimates of when sphericity is violated require
    specifying an epsilon value.

    See Muller and Barton (1989).
    http://www.jstor.org/stable/2289941?seq=1

    >>> observed_power(3,60,16)
    0.91672

    http://zoe.bme.gatech.edu/~bv20/public/samplesize.pdf
    example from page 9. 
    """
    # 05.17.2012 validated against G*Power
    # epsilon scales critical F, degrees of freedom,
    # and non-centrality parameter
    crit_f = scipy.stats.f(df*eps, dfe*eps).ppf(1.-alpha)
    return scipy.stats.ncf(df*eps, dfe*eps, nc*eps).sf(crit_f)

def f2s(L):
    """Turns a list of floats into a list of strings"""
    for i,l in enumerate(L):
        try: 
            L[i] ='%.03f'%float(l)
        except:
            pass
    return L

def matrix_rank(arr,tol=1e-8):
    """Return the matrix rank of an input array."""

    arr = np.asarray(arr)

    if len(arr.shape) != 2:
        raise ValueError('Input must be a 2-d array or Matrix object')

    svdvals = scipy.linalg.svdvals(arr)
    return sum(np.where(svdvals>tol,1,0))

def _xunique_combinations(items, n):
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in _xunique_combinations(items[i+1:],n-1):
                yield [items[i]]+cc

def _epsGG(y, df1):
    """
    (docstring is adapted from Trujillo-Ortiz (2006); see references)
    
    The Greenhouse-Geisser epsilon value measures by how much the
    sphericity assumption is violated. Epsilon is then used to adjust
    for the potential bias in the F statistic. Epsilon can be 1, which
    means that the sphericity assumption is met perfectly. An epsilon
    smaller than 1 means that the sphericity assumption is violated.
    The further it deviates from 1, the worse the violation; it can be
    as low as epsilon = 1/(k - 1), which produces the lower bound of
    epsilon (the worst case scenario). The worst case scenario depends
    on k, the number of levels in the repeated measure factor. In real
    life epsilon is rarely exactly 1. If it is not much smaller than 1,
    then we feel comfortable with the results of repeated measure ANOVA. 
    The Greenhouse-Geisser epsilon is derived from the variance-
    covariance matrix of the data. For its evaluation we need to first
    calculate the variance-covariance matrix of the variables (S). The
    diagonal entries are the variances and the off diagonal entries are
    the covariances. From this variance-covariance matrix, the epsilon
    statistic can be estimated. Also we need the mean of the entries on
    the main diagonal of S, the mean of all entries, the mean of all
    entries in row i of S, and the individual entries in the variance-
    covariance matrix. There are three important values of epsilon. It
    can be 1 when the sphericity is met perfectly. This epsilon
    procedure was proposed by Greenhouse and Geisser (1959). Greenhouse-
    Geisser's epsilon is calculated using the Satterthwaite
    approximation. See Glaser (2003.)
    
      Syntax: function _epsGG(y,df1)
    
      Inputs:
         y   = contrast matrix
         df1 = degrees of freedom of treatment
         
      Output:
         Greenhouse-Geisser epsilon value.
    
     $$We suggest you could take-a-look to the PDF document ''This Week's 
       Citation Classics'' CCNumber 28, July 12, 1982, web-page
       [http://garfield.library.upenn.edu/classics1982/A1982NW45700001.pdf]$$
    
    Example 2 of Maxwell and Delaney (p.497). This is a repeated measures
    example with two within and a subject effect. We have one dependent
    variable:reaction time, two independent variables: visual stimuli
    are tilted at 0, 4, and 8 degrees; with noise absent or present.
    Each subject responded to 3 tilt and 2 noise given 6 trials. Data are,
    
                          0           4           8                  
                     -----------------------------------
            Subject    A     P     A     P     A     P
            --------------------------------------------
               1      420   480   420   600   480   780
               2      420   360   480   480   480   600
               3      480   660   480   780   540   780
               4      420   480   540   780   540   900
               5      540   480   660   660   540   720
               6      360   360   420   480   360   540
               7      480   540   480   720   600   840
               8      480   540   600   720   660   900
               9      540   480   600   720   540   780
              10      480   540   420   660   540   780
            --------------------------------------------
    
    The three measurements of reaction time were averaging across noise 
    ausent/present. Given,
    
                             Tilt
                      -----------------
            Subject     0     4     8    
            ---------------------------
               1       450   510   630
               2       390   480   540
               3       570   630   660
               4       450   660   720
               5       510   660   630
               6       360   450   450
               7       510   600   720
               8       510   660   780
               9       510   660   660
              10       510   540   660
            ---------------------------
    
    We need to estimate the Greenhouse-Geisser epsilon associated with
    the angle of rotation of the stimuli. 
    
    Reference:
      Glaser, D.E. (2003). Variance Components. In R.S.J. Frackowiak, K.J.
          Friston, C. Firth, R. Dolan, C.J., Price, S. Zeki, J. Ashburner,
          & W.D. Penny, (Eds.), Human Brain Function. Academic Press, 2nd.
          edition. [http://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/]
      Greenhouse, S.W. and Geisser, S. (1959), On methods in the analysis
          of profile data. Psychometrika, 24:95-112. 
      Maxwell, S.E. and Delaney, H.D. (1990), Designing Experiments and 
          Analyzing Data: A model comparison perspective. Pacific Grove,
          CA: Brooks/Cole.
      Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and K.
          Barba-Rojo. (2006). _epsGG:Greenhouse-Geisser epsilon. A MATLAB
          file. [WWW document]. URL
          http://www.mathworks.com/matlabcentral/fileexchange
          /loadFile.do?objectId=12839
    """
    if df1 == 1. : return 1.
    
    V = np.cov(y) # sample covariance
    return np.trace(V)**2 / (df1*np.trace(np.dot(V.T,V)))      

def _epsHF(y, df1):
    """
    This is ported from a Matlab function written by Trujillo-Ortiz et
    al. 2006 (see references) with an important modification. If the
    calculated epsilon values is greater than 1, it returns 1. 
    
    The Huynh-Feldt epsilon its a correction of the Greenhouse-Geisser
    epsilon. This due that the Greenhouse-Geisser epsilon tends to
    underestimate epsilon when epsilon is greater than 0.70 (Stevens,
    1990). An estimated epsilon = 0.96 may be actually 1. Huynh-Feldt
    correction is less conservative. The Huynh-Feldt epsilon is
    calculated from the Greenhouse-Geisser epsilon. As the Greenhouse-
    Geisser epsilon, Huynh-Feldt epsilon measures how much the
    sphericity assumption or compound symmetry is violated. The idea of
    both corrections its analogous to pooled vs. unpooled variance
    Student's t-test: if we have to estimate more things because
    variances/covariances are not equal, then we lose some degrees of
    freedom and P-value increases. These epsilons should be 1.0 if
    sphericity holds. If not sphericity assumption appears violated.
    We must to have in mind that the greater the number of repeated
    measures, the greater the likelihood of violating assumptions of
    sphericity and normality (Keselman et al, 1996) . Therefore, we need
    to have the most conservative F values. These are obtained by
    setting epsilon to its lower bound, which represents the maximum
    violation of these assumptions. When a significant result is
    obtained, it is assumed to be robust. However, since this test may
    be overly conservative, Greenhouse and Geisser (1958, 1959)
    recommend that when the lower-bound epsilon gives a nonsignificant
    result, it should be followed by an approximate test (based on a
    sample estimate of epsilon).

      Syntax: function _epsHF(y,df1)
    
      Inputs:
         y   = contrast matrix
         df1 = degrees of freedom of treatment
         
      Output:
         Huynh-Feldt epsilon value.

    See docstring for _epsGG() for information on formatting X.
    
    Reference:
      Geisser, S, and Greenhouse, S.W. (1958), An extension of Box's
          results on the use of the F distribution in multivariate
          analysis. Annals of Mathematical Statistics, 29:885-891.
      Greenhouse, S.W. and Geisser, S. (1959), On methods in the
          analysis of profile data. Psychometrika, 24:95-112. 
      Huynh, M. and Feldt, L.S. (1970), Conditions under which mean
          square rate in repeated measures designs have exact-F
          distributions. Journal of the American Statistical
          Association, 65:1982-1989 
      Keselman, J.C, Lix, L.M. and Keselman, H.J. (1996), The analysis
          of repeated measurements: a quantitative research synthesis.
          British Journal of Mathematical and Statistical Psychology,
          49:275-298.
      Maxwell, S.E. and Delaney, H.D. (1990), Designing Experiments
          and Analyzing Data: A model comparison perspective. Pacific
          Grove, CA: Brooks/Cole.
      Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and K.
          Barba-Rojo. (2006). _epsGG:Greenhouse-Geisser epsilon. A
          MATLAB file. [WWW document].
          http://www.mathworks.com/matlabcentral/fileexchange
          /loadFile.do?objectId=12839
    """
    if df1 == 1. : return 1.
    
    k,n = np.shape(y)      # number of treatments
    eGG = _epsGG(y, df1) # Greenhouse-Geisser epsilon

    N = n*(k-1.)*eGG-2.
    D = (k-1.)*((n-1.)-(k-1.)*eGG)
    eHF = N/D                 # Huynh-Feldt epsilon estimation

    if   eHF < eGG : return eGG
    elif eHF > 1.  : return 1.
    else           : return eHF

def _epsLB(y, df1):
    """
    This is ported from a Matlab function written by Trujillo-Ortiz et
    al. 2006. See references.
    
    EPBG Box's conservative epsilon.
    The Box's conservative epsilon value (Box, 1954), measures by how
    much the sphericity assumption is violated. Epsilon is then used to
    adjust for the potential bias in the F statistic. Epsilon can be 1,
    which means that the sphericity assumption is met perfectly. An
    epsilon smaller than 1 means that the sphericity assumption is
    violated. The further it deviates from 1, the worse the violation;
    it can be as low as epsilon = 1/(k - 1), which produces the lower
    bound of epsilon (the worst case scenario). The worst case scenario
    depends on k, the number of levels in the repeated measure factor.
    In real life epsilon is rarely exactly 1. If it is not much smaller
    than 1, then we feel comfortable with the results of repeated
    measure ANOVA. The Box's conservative epsilon is derived from the
    lower bound of epsilon, 1/(k - 1). Box's conservative epsilon is no
    longer widely used. Instead, the Greenhouse-Geisser's epsilon
    represents its maximum-likelihood estimate.
    
      Syntax: function _epsLB(y,df1)
    
      Inputs:
         y   = Input matrix can be a data matrix
               (size n-data x k-treatments)
         df1 = degrees of freedom of treatment
         
      Output:
         Box's conservative epsilon value.

    See docstring for _epsGG() for information on formatting X.
    
    Reference:
      Box, G.E.P. (1954), Some theorems on quadratic forms applied in
          the study of analysis of variance problems, II. Effects of
          inequality of variance and of correlation between errors in
          the two-way classification. Annals of Mathematical Statistics.
          25:484-498. 
      Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and K.
          Barba-Rojo. (2006). _epsGG:Greenhouse-Geisser epsilon. A MATLAB
          file. [WWW document]. 
          http://www.mathworks.com/matlabcentral/fileexchange
          /loadFile.do?objectId=12839
    """
    if df1 == 1. : return 1.
        
    k = np.shape(y)[0]  # number of treatments
    box = 1./(k-1.) # Box's conservative epsilon estimation

    if box*df1 < 1. : box = 1. / df1
    
    return box
            
def windsor(X, percent):
    """
    given numpy array X returns the Windsorized trimmed samples, in
    which the trimmed values are replaced by the most extreme value
    remaining in the tail. percent should be less than 1.

    

    Example:
    
      X = array([ 3,  7, 12, 15, 17, 17, 18, 19, 19, 19,
                 20, 22, 24, 26, 30, 32, 32, 33, 36, 50])
                     
      windsor(X, .10) # trim 10% of X

      # returns trimmed array and numtrimmed
      array([12, 12, 12, 15, 17, 17, 18, 19, 19, 19,
             20, 22, 24, 26, 30, 32, 32, 33, 33, 33]), 4
    """
    X = np.array(X)
    X_copy = sorted(copy(X))
    num2exc = int(round(len(X_copy)*percent))

    if num2exc == 0:
        minval = X_copy[0]
        maxval = X_copy[-1]
    else:
        minval = X_copy[ num2exc]
        maxval = X_copy[-(num2exc+1)]
    
    X[np.where(X<minval)] = minval
    X[np.where(X>maxval)] = maxval

    return X,num2exc*2

##X=array([3,7,19,19,20,22,24,12,15,17,32,32,33,36,17,18,19,26,30,50])
##print windsor(X,.1)

class AnovaResults(OrderedDict):
    def __init__(self, *args, **kwds):

        if kwds.has_key('wfactors'):
            self.wfactors = kwds['wfactors']
        else:
            self.wfactors = []
                
        if kwds.has_key('bfactors'):
            self.bfactors = kwds['bfactors']
        else:
            self.bfactors = []

        if kwds.has_key('alpha'):
            self.alpha = kwds['alpha']
        else:
            self.alpha = 0.05

        if kwds.has_key('dv'):
            self.dv = kwds['dv']
        else:
            self.dv = None
            
        if kwds.has_key('sub'):
            self.sub = kwds['sub']
        else:
            self.sub = 'SUBJECT'

        if kwds.has_key('transform'):
            self.transform = kwds['transform']
        else:
            self.transform = ''
        
        if len(args) == 1:
            super(AnovaResults, self).__init__(args[0])
        else:
            super(AnovaResults, self).__init__()
        
    def summary(self):
        title  = '%s ~'%self.dv
        factors = self.wfactors + self.bfactors
        title += ''.join([' %s *'%f for f in factors])[:-2]

        s = [title]
        if len(self.wfactors)!=0 and len(self.bfactors)==0:
            s.append(self._within_str())
            
        if len(self.wfactors)==0 and len(self.bfactors)!=0:
            s.append(self._between_str())
            
        if len(self.wfactors)!=0 and len(self.bfactors)!=0:
            s.append(self._mixed_str())
            
        s.append(self._summary_str(factors))
        return ''.join(s)

    def _between_str(self):
        factors = self.bfactors
        
        s = ['\n\nTESTS OF BETWEEN-SUBJECTS EFFECTS\n\n']
        
        # Write ANOVA results
        s.append('Measure: %s\n'%self.dv)

        tt = TextTable(max_width=0)
        tt.set_cols_dtype(['t'] + ['a']*11)
        tt.set_cols_align(['l'] + ['r']*11)
        tt.set_deco(TextTable.HEADER | TextTable.FOOTER)
        tt.header('Source,Type III\nSS,df,MS,F,Sig.,et2_G,'
                  'Obs.,SE,95% CI,lambda,Obs.\nPower'.split(','))
        
        for i in xrange(1,len(factors)+1):
            for efs in _xunique_combinations(factors, i):
                r=self[tuple(efs)]
                src=''.join(['%s * '%f for f in efs])[:-3]
                tt.add_row([src,r['ss'],r['df'],
                            r['mss'],r['F'],r['p'],
                            r['eta'],r['obs'],r['se'],
                            r['ci'],r['lambda'],r['power']])

        tt.add_row(['Error',self[(factors[0],)]['sse'],
                   self[(factors[0],)]['dfe'],
                   self[(factors[0],)]['mse'],
                   '','','','','','','',''])

        ss_total = np.sum((self.df[self.dv] - np.mean(self.df[self.dv]))**2)
        df_total = len(self.df[self.dv]) - 1 - self.dftrim
        
        tt.footer(['Total',ss_total,df_total,
                   '','','','','','','','',''])
        
        s.append(tt.draw())
        return ''.join(s)

    def _mixed_str(self):
        bfactors=self.bfactors
        wfactors=self.wfactors
        factors=wfactors+bfactors
        df = self.df
        
        # Write Tests of Between-Subjects Effects

        s = ['\n\nTESTS OF BETWEEN-SUBJECTS EFFECTS\n\n']
        
        # Write ANOVA results
        s.append('Measure: %s\n'%self.dv)

        tt = TextTable(max_width=0)
        tt.set_cols_dtype(['t'] + ['a']*11)
        tt.set_cols_align(['l'] + ['r']*11)
        tt.set_deco(TextTable.HEADER | TextTable.FOOTER)
        tt.header('Source,Type III\nSS,df,MS,F,Sig.,et2_G,'
                  'Obs.,SE,95% CI,lambda,Obs.\nPower'.split(','))

        tt.add_row(['Between Subjects',
                    self[(self.sub,)]['ss'],
                    self[(self.sub,)]['df'],
                    '','','','','','','','',''])
        
        for i in xrange(1,len(bfactors)+1):
            for efs in _xunique_combinations(bfactors, i):
                r=self[tuple(efs)]
                src=''.join(['%s * '%f for f in efs])[:-3]
                tt.add_row([src,r['ss'],r['df'],
                            r['mss'],r['F'],r['p'],
                            r['eta'],r['obs'],r['se'],
                            r['ci'],r['lambda'],r['power']])

        tt.footer(['Error',
                   self[(self.sub,)]['sse'],
                   self[(self.sub,)]['dfe'],
                   self[(self.sub,)]['mse'],
                   '','','','','','','',''])
        s.append(tt.draw())
        
    
        # Write Tests of Within-Subjects Effects

        s.append('\n\nTESTS OF WITHIN SUBJECTS EFFECTS\n\n')
        
        # Write ANOVA 
        s.append('Measure: %s\n'%self.dv)

        tt = TextTable(max_width=0)
        tt.set_cols_dtype(['t']*2 + ['a']*12)
        tt.set_cols_align(['l']*2 + ['r']*12)
        tt.set_deco(TextTable.HEADER | TextTable.HLINES)
        tt.header('Source,,Type III\nSS,eps,df,MS,F,Sig.,'
                  'et2_G,Obs.,'
                  'SE,95% CI,lambda,Obs.\nPower'.split(','))
        
        defs=[]
        for i in xrange(1,len(wfactors)+1):
            for efs in _xunique_combinations(wfactors, i):
                defs.append(efs)
                treatment = []
                r=self[tuple(efs)]
                src=' *\n'.join(efs)
                treatment.append([src,'Sphericity Assumed',
                   r['ss'],' - ',r['df'],r['mss'],r['F'],r['p'],
                   r['eta'],r['obs'],r['se'],
                   r['ci'],r['lambda'],r['power']])
                treatment.append(['', 'Greenhouse-Geisser',
                   r['ss'],r['eps_gg'],r['df_gg'],r['mss_gg'],r['F_gg'],
                   r['p_gg'],r['eta'],r['obs_gg'],r['se_gg'],
                   r['ci_gg'],r['lambda_gg'],r['power_gg']])
                treatment.append(['', 'Huynh-Feldt',
                   r['ss'],r['eps_hf'],r['df_hf'],r['mss_hf'],r['F_hf'],
                   r['p_hf'],r['eta'],r['obs_hf'],r['se_hf'],
                   r['ci_hf'],r['lambda_hf'],r['power_hf']])
                treatment.append(['', 'Box',
                   r['ss'],r['eps_lb'],r['df_lb'],r['mss_lb'],r['F_lb'],
                   r['p_lb'],r['eta'],r['obs_lb'],r['se_lb'],
                   r['ci_lb'],r['lambda_lb'],r['power_lb']])

                row = []
                for i in _xrange(14):
                    row.append('\n'.join(_str(treatment[j][i])
                                         for j in _xrange(4)))
                tt.add_row(row)
                
                for i in xrange(1,len(factors)+1):
                    for efs2 in _xunique_combinations(factors, i):
                        if efs2 not in self.befs and \
                           efs2 not in defs and \
                           efs2 not in self.wefs \
                           and len(set(efs2).difference(set(efs+bfactors)))==0:
                            defs.append(efs2)
                            treatment = []
                            r=self[tuple(efs2)]
                            src=''.join(['%s * '%f for f in efs2])[:-3]
                            treatment.append([src,'Sphericity Assumed',
                               r['ss'],' - ',r['df'],r['mss'],r['F'],r['p'],
                               r['eta'],r['obs'],r['se'],
                               r['ci'],r['lambda'],r['power']])
                            treatment.append(['', 'Greenhouse-Geisser',
                               r['ss'],r['eps_gg'],r['df_gg'],r['mss_gg'],
                               r['F_gg'],r['p_gg'],r['eta'],r['obs_gg'],
                               r['se_gg'],r['ci_gg'],
                               r['lambda_gg'],r['power_gg']])
                            treatment.append(['', 'Huynh-Feldt',
                               r['ss'],r['eps_hf'],r['df_hf'],r['mss_hf'],
                               r['F_hf'],r['p_hf'],r['eta'],r['obs_hf'],
                               r['se_hf'],r['ci_hf'],
                               r['lambda_hf'],r['power_hf']])
                            treatment.append(['', 'Box',
                               r['ss'],r['eps_lb'],r['df_lb'],r['mss_lb'],
                               r['F_lb'],r['p_lb'],r['eta'],r['obs_lb'],
                               r['se_lb'],r['ci_lb'],
                               r['lambda_lb'],r['power_lb']])
                            
                            row = []
                            for i in _xrange(14):
                                row.append('\n'.join(_str(treatment[j][i])
                                                     for j in _xrange(4)))
                            tt.add_row(row)
                            
                error = []
                
                src='Error(%s)'%' *\n'.join([f for f in efs if
                                             f not in bfactors])
                error.append([src,'Sphericity Assumed',
                   r['sse'],' - ',r['dfe'],r['mse'],
                   '','','','','','','',''])
                error.append(['', 'Greenhouse-Geisser',
                   r['sse'],r['eps_gg'],r['dfe_gg'],r['mse_gg'],
                   '','','','','','','',''])
                error.append(['', 'Huynh-Feldt',
                   r['sse'],r['eps_hf'],r['dfe_hf'],r['mse_hf'],
                   '','','','','','','',''])
                error.append(['', 'Box',
                   r['sse'],r['eps_lb'],r['dfe_lb'],r['mse_lb'],
                   '','','','','','','',''])

                row = []
                for i in _xrange(14):
                    row.append('\n'.join(_str(error[j][i])
                                         for j in _xrange(4)))
                tt.add_row(row)

        s.append(tt.draw())
        return ''.join(s)
    
    def _within_str(self):
        factors=self.wfactors
        
        s = ['\n\nTESTS OF WITHIN SUBJECTS EFFECTS\n\n']
        
        # Write ANOVA 
        s.append('Measure: %s\n'%self.dv)

        tt = TextTable(max_width=0)
        tt.set_cols_dtype(['t']*2 + ['a']*12)
        tt.set_cols_align(['l']*2 + ['r']*12)
        tt.set_deco(TextTable.HEADER | TextTable.HLINES)
        tt.header('Source,,Type III\nSS,eps,df,MS,F,Sig.,'
                  'et2_G,Obs.,'
                  'SE,95% CI,lambda,Obs.\nPower'.split(','))
        
        for i in xrange(1,len(factors)+1):
            for efs in _xunique_combinations(factors, i):
                treatment = []
                r=self[tuple(efs)]
                src=' *\n'.join(efs)
                treatment.append([src,'Sphericity Assumed',
                   r['ss'],' - ',r['df'],r['mss'],r['F'],r['p'],
                   r['eta'],r['obs'],r['se'],
                   r['ci'],r['lambda'],r['power']])
                treatment.append(['', 'Greenhouse-Geisser',
                   r['ss'],r['eps_gg'],r['df_gg'],r['mss_gg'],r['F_gg'],
                   r['p_gg'],r['eta'],r['obs_gg'],r['se_gg'],
                   r['ci_gg'],r['lambda_gg'],r['power_gg']])
                treatment.append(['', 'Huynh-Feldt',
                   r['ss'],r['eps_hf'],r['df_hf'],r['mss_hf'],r['F_hf'],
                   r['p_hf'],r['eta'],r['obs_hf'],r['se_hf'],
                   r['ci_hf'],r['lambda_hf'],r['power_hf']])
                treatment.append(['', 'Box',
                   r['ss'],r['eps_lb'],r['df_lb'],r['mss_lb'],r['F_lb'],
                   r['p_lb'],r['eta'],r['obs_lb'],r['se_lb'],
                   r['ci_lb'],r['lambda_lb'],r['power_lb']])

                row = []
                for i in _xrange(14):
                    row.append('\n'.join(_str(treatment[j][i])
                                         for j in _xrange(4)))
                tt.add_row(row)

                error = []
                src='Error(%s)'%src
                error.append([src,'Sphericity Assumed',
                   r['sse'],' - ',r['dfe'],r['mse'],
                   '','','','','','','',''])
                error.append(['', 'Greenhouse-Geisser',
                   r['sse'],r['eps_gg'],r['dfe_gg'],r['mse_gg'],
                   '','','','','','','',''])
                error.append(['', 'Huynh-Feldt',
                   r['sse'],r['eps_hf'],r['dfe_hf'],r['mse_hf'],
                   '','','','','','','',''])
                error.append(['', 'Box',
                   r['sse'],r['eps_lb'],r['dfe_lb'],r['mse_lb'],
                   '','','','','','','',''])

                row = []
                for i in _xrange(14):
                    row.append('\n'.join(_str(error[j][i])
                                         for j in _xrange(4)))
                tt.add_row(row)

        s.append(tt.draw())

        return ''.join(s)

    def _summary_str(self, factors):
        
        # Write Summary Means
        s = ['\n\nTABLES OF ESTIMATED MARGINAL MEANS\n\n']
        for i in xrange(1,len(factors)+1):
            for efs in _xunique_combinations(factors, i):
                s.append('Estimated Marginal Means for ' + \
                     ''.join(['%s * '%f for f in efs])[:-3] + '\n')

                dave = pandas.pivot_table(self.df, values=self.dv, rows=efs)
                names = dave.index
                dave = np.array(dave).flatten()
                
                dsem = pandas.pivot_table(self.df, values=self.dv, rows=efs,
                                          aggfunc=lambda x : scipy.stats.sem(x, axis=None))
                dsem = np.array(dsem).flatten()
                
                dlowr=dave-(dsem*1.96)
                dhghr=dave+(dsem*1.96)

                tt = TextTable(max_width=0)
                tt.set_cols_dtype(['t']*len(efs) + ['a','a','a','a'])
                tt.set_cols_align(['l']*len(efs) + ['r','r','r','r'])
                tt.set_deco(TextTable.HEADER)
                
                tt.header(efs+['Mean','Std. Error',
                               '95% Lower Bound',
                               '95% Upper Bound'])

                if isinstance(names, pandas.MultiIndex):
                    for i,name in enumerate(names):
                        tt.add_row(list(name)+[dave[i],dsem[i],dlowr[i],dhghr[i]])
                else:
                    for i,name in enumerate(names):
                        tt.add_row([name]+[dave[i],dsem[i],dlowr[i],dhghr[i]])
                    

                s.append(tt.draw())
                s.append('\n\n')

        return ''.join(s)
                
##    def plot(self, val, xaxis,
##             seplines=None,
##             sepxplots=None,
##             sepyplots=None,
##             xmin='AUTO',xmax='AUTO',
##             ymin='AUTO',ymax='AUTO',
##             fname=None,
##             quality='low',
##             errorbars='ci',
##             output_dir=''):
##        """
##        This functions is basically wraps the plot function from the
##        dataframe module. It attempts to find the appropriate error bar
##        term. Creats a filename if necessary and calls plot.
##        """
##
##
##        # Attempt to find errorbars
##        factors=self.wfactors+self.bfactors
##        efs=[f for f in factors if f in [xaxis,seplines,sepxplots,sepyplots]]
##
##        if errorbars=='ci':
##            if len(self.wfactors)==0 and len(self.bfactors)!=0:
##                yerr=self[tuple(efs)]['ci']
##            else:
##                yerr=self[tuple(efs)]['ci_gg']
##        elif errorbars=='sem':
##            if len(self.wfactors)==0 and len(self.bfactors)!=0:
##                yerr=self[tuple(efs)]['se']
##            else:
##                yerr=self[tuple(efs)]['se_gg']
##        else:
##            yerr=None
##
##        # turn on TESTMODE in the DataFrame so we can get the filename
##        self.df.TESTMODE = True
##        
##        D = self.df.interaction_plot(val, xaxis, seplines=seplines,
##                                 sepxplots=sepxplots,
##                                 sepyplots=sepyplots,
##                                 xmin=xmin, xmax=xmax,
##                                 ymin=ymin, ymax=ymax,
##                                 fname=fname,
##                                 quality=quality,
##                                 yerr=yerr,
##                                 output_dir=output_dir)
##        return D

    def __repr__(self):
        if self == {}:
            return 'AnovaResults()'

        s = []
        for k, v in self.items():
            s.append("(%s, %s)"%(repr(k), repr(v)))
        args = '[' + ', '.join(s) + ']'
        
        kwds = []
        
##        kwds.append(", df=%s"%repr(self.df))
            
        if self.wfactors != []:
            kwds.append(", wfactors=%s"%repr(self.wfactors))
                
        if self.bfactors != []:
            kwds.append(", bfactors=%s"%repr(self.bfactors))

        if self.alpha != 0.05:
            kwds.append(", alpha=%s"%self.alpha)

        if self.dv != None:
            kwds.append(", dv='%s'"%self.dv)
            
        if self.sub != 'SUBJECT':
            kwds.append(", sub='%s'"%self.sub)

        if self.transform != '':
            kwds.append(", transform='%s'"%self.transform)
                
        kwds= ''.join(kwds)
        
        return 'Anova(%s%s)'%(args,kwds)

def _between(results_obj):
    factors=results_obj.bfactors
    pt_asarray = results_obj.pt_asarray
    D = results_obj.D
    numlevels = results_obj.numlevels
    df = results_obj.df
    
    Nf = len(D)      # Number of factors
    Nd = np.prod(D)  # Total number of conditions
    Ne = 2**Nf - 1   # Number of effects
    Nr,Nn = np.shape(pt_asarray) # Number of replications (eg subjects)
                              # x treatments

    if np.shape(pt_asarray)[1] != Nd:
        raise Exception('data has %d conditions; design only %d',
                        np.shape(pt_asarray)[1],Nd)
    
    sc, sy = {}, {}
    for f in xrange(1,Nf+1):
        # create main effect/interaction component contrasts
        sc[(f,1)] = np.ones((D[f-1],1))
        sc[(f,2)] = scipy.signal.detrend(np.eye(D[f-1]),type='constant')

        # create main effect/interaction components for means
        sy[(f,1)] = np.ones((D[f-1],1))/D[f-1]
        sy[(f,2)] = np.eye(D[f-1])

    # Loop through effects
    # Do fancy calculations
    # Record the results of the important fancy calcuations
    for e in xrange(1,Ne+1):
        
        # code effects so we can build contrasts
        cw = _num2binvec(e,Nf)
        efs = np.asarray(factors)[Nf-1-np.where(np.asarray(cw)==2.)[0][::-1]]
        r = {}
    
        # create full contrasts
        c = sc[(1,cw[Nf-1])];  
        for f in xrange(2,Nf+1):
            c = np.kron(c, sc[(f,cw[Nf-f])])
            
        Nc = np.shape(c)[1]  # Number of conditions in effect
        No = Nd/Nc*1.   # Number of observations per condition in effect
        
        # project data to contrast sub-space
        y  = np.dot(pt_asarray,c)
        nc = np.shape(y)[1] 
        
        # calculate component means
        cy = sy[(1, cw[Nf-1])]
        for f in xrange(2,Nf+1):
            cy = np.kron(cy, sy[(f,cw[Nf-f])] )
    
        r['y2'] = np.mean(np.dot(pt_asarray,cy),0)
        
        # calculate df, ss, and mss
        b = np.mean(y,0)
        r['df'] = float(matrix_rank(c))
        r['ss'] = np.sum(y*b.T)*Nc
        r['mss'] = r['ss']/r['df']

        results_obj[tuple(efs)]=r
    
    ss_total = np.sum((pt_asarray - np.mean(pt_asarray))**2)
    ss_error = ss_total
    dfe=len(results_obj.df[results_obj.dv]) - 1. - results_obj.dftrim
    
    for i in xrange(1,len(factors)+1):
        for efs in _xunique_combinations(factors, i):
            ss_error -= results_obj[tuple(efs)]['ss']
            dfe -= results_obj[tuple(efs)]['df']

    # calculate F, p, and standard errors
    for i in xrange(1,len(factors)+1):
        for efs in _xunique_combinations(factors, i):

            r = results_obj[tuple(efs)]

            r['sse'] = ss_error
            r['dfe'] = dfe
            r['mse'] = ss_error / dfe
            r['F'] = r['mss']/r['mse']
            r['p'] = scipy.stats.f(r['df'],r['dfe']).sf(r['F'])

            # calculate Generalized eta effect size
            r['eta'] = r['ss']/(r['ss']+r['sse'])

            # calculate observations per cell
            r['obs']  = numlevels[results_obj.sub]
            r['obs'] /= np.prod([numlevels[f]*1. for f in efs])
            
            # calculate Loftus and Masson standard errors
            r['critT'] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
            r['se'] = np.sqrt(r['mse']/r['obs'])*r['critT']/1.96
            r['ci'] = np.sqrt(r['mse']/r['obs'])*r['critT']

            # calculate non-centrality and observed power
            r['lambda'] = noncentrality_parameter(r['ss'], r['sse'], r['obs'])
            r['power'] = observed_power( r['df'], r['dfe'], r['lambda'] )
            
            # record to dict
            results_obj[tuple(efs)] = r
        
def _mixed(results_obj):
    ## Programmer note:
    ## The order of in which things happen is extremely critical.
    ## Use extreme caution when modifying this function.
    
    bfactors = results_obj.bfactors
    wfactors = results_obj.wfactors
    factors = wfactors+bfactors
    pt_asarray = results_obj.pt_asarray
    D = results_obj.D
    numlevels = results_obj.numlevels
    df = results_obj.df
    
    Nf = len(D)      # Number of factors
    Nd = np.prod(D)     # Total number of conditions
    Ne = 2**Nf - 1   # Number of effects
    Nr,Nn = np.shape(pt_asarray) # Number of replications (eg subjects)
                              # x treatments

    if np.shape(pt_asarray)[1] != Nd:
        raise Exception('data has %d conditions; design only %d',
                        np.shape(pt_asarray)[1],Nd)
    
    sc,sy = {},{}
    for f in xrange(1,Nf+1):
        # create main effect/interaction component contrasts
        sc[(f,1)] = np.ones((D[f-1],1))
        sc[(f,2)] = scipy.signal.detrend(np.eye(D[f-1]),type='constant')

        # create main effect/interaction components for means
        sy[(f,1)] = np.ones((D[f-1],1))/D[f-1]
        sy[(f,2)] = np.eye(D[f-1])

    # Loop through effects
    # Do fancy calculations
    # Record the results of the important fancy calcuations
    for e in xrange(1,Ne+1):

        # code effects so we can build contrasts
        cw = _num2binvec(e,Nf)
        efs = np.asarray(factors)[Nf-1-np.where(np.asarray(cw)==2.)[0][::-1]]
        r={}
    
        # create full contrasts
        c = sc[(1,cw[Nf-1])];  
        for f in xrange(2,Nf+1):
            c = np.kron(c, sc[(f,cw[Nf-f])])
            
        Nc = np.shape(c)[1] # Number of conditions in effect
        No = Nd/Nc*1.   # Number of observations per condition in effect

        # project data to contrast sub-space
        y = np.dot(pt_asarray,c)

        # calculate component means
        cy = sy[(1, cw[Nf-1])]
        for f in xrange(2,Nf+1):
            cy = np.kron(cy, sy[(f,cw[Nf-f])] )
    
        r['y2'] = np.mean(np.dot(pt_asarray,cy),0)
        
        # df for effect
        r['df'] =  np.prod([numlevels[f]-1. for f in efs])

        # calculate Greenhouse-Geisser & Huynh-Feldt epsilons
        r['eps_gg'] = _epsGG(y, r['df'])
        r['eps_hf'] = _epsHF(y, r['df'])
        r['eps_lb'] = _epsLB(y, r['df'])
        
        # calculate ss, sse, mse, mss, F, p, and standard errors
        b = np.mean(y,0)
        
        # Sphericity assumed
        r['ss']  = np.sum(y*b.T)
        r['ss'] /= No/(np.prod([numlevels[f] for f in bfactors])*1.)
        r['mss'] = r['ss']/r['df']
        
        results_obj[tuple(efs)] = r

    # calculate sse, dfe, and mse for between subjects effects
    dfe_sum   = 0. # for df trim
    ss_total  = np.sum((pt_asarray - np.mean(pt_asarray))**2)

    sub_means = pandas.pivot_table(results_obj.df,
                                   values=results_obj.dv,
                                   rows=[results_obj.sub])
    
    sub_means = np.array(sub_means, dtype=np.float64)
    
    ss_bsub  =  np.sum((sub_means - np.mean(pt_asarray))**2)
    ss_bsub *= (np.prod([numlevels[f] for f in wfactors])*1.)

    df_b  = np.prod([numlevels[f] for f in bfactors])
    df_b *= Nr/np.prod([numlevels[f] for f in bfactors])
    df_b -= 1.

    dfe_b  =  np.prod([numlevels[f] for f in bfactors])
    dfe_b *= (Nr/np.prod([numlevels[f] for f in bfactors])-1.)
    
    dfe_sum += dfe_b


    df_total = Nr/np.prod([numlevels[f]*1. for f in bfactors])*Nd-1
    
    sse_b = ss_bsub 
    results_obj.befs = [] # list of between subjects effects
    for i in xrange(1,len(bfactors)+1):
        for efs in _xunique_combinations(bfactors, i):
            sse_b -= results_obj[tuple(efs)]['ss']
            results_obj.befs.append(efs) 

    mse_b=sse_b/dfe_b

    # store calculations to reuslts dictionary
    results_obj[(results_obj.sub,)] = {'ss'  : ss_bsub,
                                       'sse' : sse_b,
                                       'mse' : mse_b,
                                       'df'  : df_b,
                                       'dfe' : dfe_b}

    results_obj[('TOTAL',)] = {'ss' : ss_total,
                               'df' : df_total}
    results_obj[('WITHIN',)] = {'ss' : ss_total-ss_bsub,
                                'df' : results_obj[('TOTAL',)]['df']-df_b}

    ss_err_tot=0.
    
    # calculate ss, df, and mss for within subjects effects
    results_obj.wefs=[]
    for i in xrange(1, len(wfactors)+1):
        for efs in _xunique_combinations(wfactors, i):
                
            results_obj.wefs.append(efs)
            efs+=[results_obj.sub]
            
            r={}
            tmp = pandas.pivot_table(results_obj.df,
                                     values=results_obj.dv,
                                     rows=[results_obj.sub])                          
            tmp = np.array(tmp, dtype=np.float64).flatten()
            r['ss']  = np.sum((tmp - np.mean(pt_asarray))**2)
            r['ss'] *= np.prod([numlevels[f] for f in wfactors
                             if f not in efs])

            for j in xrange(1, len(efs+bfactors)+1):
                for efs2 in _xunique_combinations(efs+bfactors, j):
                    if efs2 not in results_obj.befs and efs2!=efs:
                        if not ( results_obj.sub in efs2 and 
                                 len(set(efs2).intersection(set(bfactors))) > 0):
                            r['ss'] -= results_obj[tuple(efs2)]['ss']
                            
            ss_err_tot+=r['ss']
            
            r['df']  = np.prod([numlevels[f] for f in bfactors])
            r['df'] *= np.prod([numlevels[f]-1. for f in efs \
                             if f in wfactors])
            r['df'] *= Nr/np.prod([numlevels[f]*1. for f in bfactors])-1.
            dfe_sum += r['df']

            r['mss'] = r['ss']/r['df']
      
            results_obj[tuple(efs)]=r

    # trim df for between subjects effects
    results_obj[(results_obj.sub,)]['dfe'] = dfe_b - (dfe_b/dfe_sum) * results_obj.dftrim
    ss_err_tot+=mse_b*dfe_b
    
    # calculate mse, dfe, sse, F, p, and standard errors
    # between subjects effects        
    for i in xrange(1,len(bfactors)+1):
        for efs in _xunique_combinations(bfactors, i):
            r = results_obj[tuple(efs)]
            
            r['sse'] = mse_b*dfe_b
            r['dfe'] = results_obj[(results_obj.sub,)]['dfe']
            r['mse'] = r['sse']/r['dfe']
            r['F'] = r['mss']/r['mse']
            r['p'] = scipy.stats.f(r['df'],r['dfe']).sf(r['F'])
            
            # calculate Generalized eta effect size 
            r['eta'] = r['ss']/(r['ss']+ss_err_tot)

            # calculate observations per cell
            r['obs'] = numlevels[results_obj.sub]
            r['obs']/= np.prod([numlevels[f]*1. for f in efs])

            # calculate Loftus and Masson standard errors
            r['critT'] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
            r['se'] = np.sqrt(r['mse']/r['obs'])*r['critT']/1.96
            r['ci'] = np.sqrt(r['mse']/r['obs'])*r['critT']

            # calculate non-centrality and observed power
            r['lambda'] = noncentrality_parameter(r['ss'], r['sse'], r['obs'])
            r['power'] = observed_power( r['df'], r['dfe'], r['lambda'] )
            
            # record to dict
            results_obj[tuple(efs)] = r                
    
    # calculate mse, dfe, sse, F, p, and standard errors
    # within subjects effects
    for i in xrange(1,len(factors)+1):
        for efs in _xunique_combinations(factors, i):
            
            if efs not in results_obj.befs:
                r=results_obj[tuple(efs)]
                r2=results_obj[tuple([f for f in efs if f not in bfactors] +
                              [results_obj.sub])]

                r['dfe'] = r2['df'] - (r2['df']/dfe_sum) * results_obj.dftrim
                r['sse'] = r2['ss']
                r['mse'] = r2['mss']
                r['F'] = r['mss']/r['mse']
                r['p'] = scipy.stats.f(r['df'],r['dfe']).sf(r['F'])
                
                # calculate Generalized eta effect size 
                r['eta'] = r['ss']/(r['ss']+ss_err_tot)

                # calculate observations per cell
                r['obs'] = Nr/np.prod([numlevels[f]*1. for f in bfactors])
                r['obs'] *= np.prod([numlevels[f]*1. for f in factors])
                r['obs'] /= np.prod([numlevels[f]*1. for f in efs])

                # calculate Loftus and Masson standard errors
                r['critT'] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
                r['se'] = np.sqrt(r['mse']/r['obs'])*r['critT']/1.96
                r['ci'] = np.sqrt(r['mse']/r['obs'])*r['critT']

                # calculate non-centrality and observed power
                r['lambda'] = noncentrality_parameter(r['ss'], r['sse'], r['obs'])
                r['power']=observed_power( r['df'], r['dfe'], r['lambda'] )

                # Greenhouse-Geisser, Huynh-Feldt, Lower-Bound
                for x in ['_gg','_hf','_lb']:
                    r['df%s'%x] = r['df']*r['eps%s'%x]
                    r['dfe%s'%x] = r['dfe']*r['eps%s'%x]
                    r['mss%s'%x] = r['ss']/r['df%s'%x]
                    r['mse%s'%x] = r['sse']/r['dfe%s'%x]
                    r['F%s'%x] = r['mss%s'%x]/r['mse%s'%x]
                    r['p%s'%x] = scipy.stats.f(r['df%s'%x],r['dfe%s'%x]).sf(r['F%s'%x])
                    r['obs%s'%x] = r['obs']
                    r['critT%s'%x] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
                    r['se%s'%x] = np.sqrt(r['mse']/r['obs%s'%x])*\
                                     r['critT%s'%x]/1.96
                    r['ci%s'%x] = np.sqrt(r['mse']/r['obs%s'%x])*\
                                     r['critT%s'%x]

                    # calculate non-centrality and observed power
                    r['lambda%s'%x]=r['lambda']
                    r['power%s'%x]=observed_power( r['df'], r['dfe'], r['lambda'],
                                                   eps=r['eps%s'%x])

                # record to dict
                results_obj[tuple(efs)]=r
           
def _within(results_obj):
    factors = results_obj.wfactors
    pt_asarray = results_obj.pt_asarray
    D = results_obj.D
    numlevels = results_obj.numlevels
    df = results_obj.df
    
    Nf = len(D)      # Number of factors
    Nd = np.prod(D)     # Total number of conditions
    Ne = 2**Nf - 1   # Number of effects
    Nr,Nn = np.shape(pt_asarray) # Number of replications (eg subjects)
                              # x treatments

    if np.shape(pt_asarray)[1] != Nd:
        raise Exception('data has %d conditions; design only %d',
                        np.shape(pt_asarray)[1],Nd)
    
    sc,sy = {},{}
    for f in xrange(1,Nf+1):
        # create main effect/interaction component contrasts
        sc[(f,1)] = np.ones((D[f-1],1))
        sc[(f,2)] = scipy.signal.detrend(np.eye(D[f-1]),type='constant')

        # create main effect/interaction components for means
        sy[(f,1)] = np.ones((D[f-1],1))/D[f-1]
        sy[(f,2)] = np.eye(D[f-1])
        
    # Calulate dfs and dfes
    dfe_sum=0.
    for i in xrange(1,len(factors)+1):
        for efs in _xunique_combinations(factors, i):
            r={}
            
            r['df']  = np.prod([numlevels[f]-1 for f in factors if \
                             f in efs])
            r['dfe'] = float(r['df']*(Nr-1.))            
            dfe_sum  += r['dfe'] # for df trim

            results_obj[tuple(efs)]=r

    # Loop through effects
    # Do fancy calculations
    # Record the results of the important fancy calcuations
    for e in xrange(1,Ne+1):
        # code effects so we can build contrasts
        cw = _num2binvec(e,Nf)
        efs = np.asarray(factors)[Nf-1-np.where(np.asarray(cw)==2.)[0][::-1]]
        r=results_obj[tuple(efs)] # unpack dictionary
    
        # create full contrasts
        c = sc[(1,cw[Nf-1])];  
        for f in xrange(2,Nf+1):
            c = np.kron(c, sc[(f,cw[Nf-f])])
            
        Nc = np.shape(c)[1] # Number of conditions in effect
        No = Nd/Nc*1.   # Number of observations per condition in effect
        
        # project data to contrast sub-space   
        y  = np.dot(pt_asarray, c)
        nc = np.shape(y)[1]

        # calculate component means
        cy = sy[(1, cw[Nf-1])]
        for f in xrange(2,Nf+1):
            cy = np.kron(cy, sy[(f,cw[Nf-f])] )
    
        r['y2'] = np.mean(np.dot(pt_asarray,cy),0)
        
        # calculate Greenhouse-Geisser & Huynh-Feldt epsilons
        r['eps_gg'] = _epsGG(y, r['df'])
        r['eps_hf'] = _epsHF(y, r['df'])
        r['eps_lb'] = _epsLB(y, r['df'])
        
        # calculate ss, sse, mse, mss, F, p, and standard errors
        b = np.mean(y,0)
        
        # Sphericity assumed
        r['dfe'] -= (r['dfe']/dfe_sum) * results_obj.dftrim
        r['ss']   =  np.sum(y*b.T)
        r['mse']  = (np.sum(np.diag(np.dot(y.T,y)))-r['ss'])/r['dfe']
        r['sse']  =  r['dfe']*r['mse']

        r['ss'] /=  No
        r['mss'] =  r['ss']/r['df']
        r['sse']/=  No
        r['mse'] =  r['sse']/r['dfe']
        
        r['F'] =  r['mss']/r['mse']
        r['p'] =  scipy.stats.f(r['df'],r['dfe']).sf(r['F'])
        
        # calculate observations per cell
        r['obs'] =  Nr*No

        # calculate Loftus and Masson standard errors
        r['critT'] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
        r['se'] = np.sqrt(r['mse']/r['obs'])*r['critT']/1.96
        r['ci'] = np.sqrt(r['mse']/r['obs'])*r['critT']

        # calculate non-centrality and observed power
        r['lambda'] = noncentrality_parameter(r['ss'], r['sse'], r['obs'])
        r['power'] = observed_power(r['df'], r['dfe'], r['lambda'])

        # Greenhouse-Geisser, Huynh-Feldt, Lower-Bound
        for x in ['_gg','_hf','_lb']:
            r['df%s'%x]  = r['df']*r['eps%s'%x]
            r['dfe%s'%x] = r['dfe']*r['eps%s'%x]
            r['mss%s'%x] = r['ss']/r['df%s'%x]
            r['mse%s'%x] = r['sse']/r['dfe%s'%x]
            r['F%s'%x] = r['mss%s'%x]/r['mse%s'%x]
            r['p%s'%x] = scipy.stats.f(r['df%s'%x],r['dfe%s'%x]).sf(r['F%s'%x])
            r['obs%s'%x] = Nr*No
            r['critT%s'%x] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
            r['se%s'%x] = np.sqrt(r['mse']/r['obs%s'%x])*r['critT%s'%x]/1.96
            r['ci%s'%x] = np.sqrt(r['mse']/r['obs%s'%x])*r['critT%s'%x]

            # calculate non-centrality and observed power
            r['lambda%s'%x]=r['lambda']
            r['power%s'%x]=observed_power( r['df'], r['dfe'], r['lambda'] , eps=r['eps%s'%x])
                        
        # record to dict
        results_obj[tuple(efs)]=r

    # Calculate parameters need to calculate effect size estimates
    sub_means   =  np.mean(pt_asarray, axis=1)
    ss_subject  =  np.sum((sub_means - np.mean(pt_asarray))**2)
    ss_subject *= (np.prod([numlevels[f] for f in factors])*1.)
    ss_err_tot  =  sum([r['sse'] for r in results_obj.values()])

    # Loop through and calculate Generalize eta effect sizes
    for efs,r in results_obj.items():        
        r['eta']   = r['ss']/(ss_subject + ss_err_tot)
        results_obj[tuple(efs)]=r

def _num2binvec(d,p=0):
    """Sub-function to code all main effects/interactions"""
    d,p=float(d),float(p)
    d=abs(round(d))

    if d==0.:
        b=0.
    else:
        b=[]
        while d>0.:
            b.insert(0,float(np.remainder(d,2.)))
            d=np.floor(d/2.)

    return list(np.array(list(np.zeros((p-len(b))))+b)+1.)
   
def anova(dataframe, dv, wfactors=None, bfactors=None,
          sub='SUBJECT', transform='', alpha=0.05):  
    """
    single or multiple factor between, within, and mixed ANOVA

    Parameters
    ----------
    dataframe : a pandas.DataFrame object with records in the stacked
        (tall) format
    dv : dependent variable to analyze. Should have interval or ratio
        scaling
        
    **kwargs**
    
    wfactors : list of within factors or None
    bfactors : list of between factors or None
    sub : isomorphism variable (subjects variable)
    transform: string specifying a data transformation
     =======================  ===============  ==================  
     STRING OPTION            TRANSFORM        COMMENTS
     =======================  ===============  ==================
     ''                       X                default
     'log','log10'            numpy.log(X)     base 10 transform
     'reciprocal', 'inverse'  1/X
     'square-root', 'sqrt'    numpy.sqrt(X)
     'arcsine', 'arcsin'      numpy.arcsin(X)
     'windsor 10'             windsor(X, 10)   10% windosr trim
     =======================  ===============  ==================
     
    Notes
    -----
    wfactors and bfactors kwargs cannot both be None.

    Greenhouse-Geisser's epsilon is calculated using the
    Satterthwaite approximation. See Glaser (2003.)

    Generalized eta squared is a measure of effect size which
    allows comparibility across between-subjects and
    within-subjects designs. For background and derivation
    see Olejnik and Algina 2003. For further information and
    endorsement see Bakeman 2005.

    Standard Errors and 95% confidence intervals are calculated
    according to Loftus and Masson (1994).

    The details and rational for calculating the
    non-centrality paramenter can be found:
        http://epm.sagepub.com/content/55/6/998.full.pdf+html
        http://www.jstor.org/stable/3701269?seq=2
        http://zoe.bme.gatech.edu/~bv20/public/samplesize.pdf
        http://www.jstor.org/stable/2289941?seq=1

    References:
      Glaser, D.E. (2003). Variance Components. In R.S.J. Frackowiak, K.J.
          Friston, C. Firth, R. Dolan, C.J., Price, S. Zeki, J. Ashburner,
          & W.D. Penny, (Eds.), Human Brain Function. Academic Press, 2nd.
          edition. [http://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/]
      Hopkins, K. D., & Hester, P. R. (1995). The noncentrality parameter
          for the F distribution, mean squares, and effect size: An
          Examination of some mathematical relationshipss. Educational and
          Psychological Measurement, 55 (6), 998-999.
      Howell, D.C. (2001). Statistical Methods for Psychology. Wadsworth
          Publishing, CA.
      Liu, X., & Raudenbush, S. (2004). A note on the noncentrality parameter
          and effect size estimates for the F Test in ANOVA. Journal of Ed.
          and Behavioral Statistics, 29 (2), 251-255.
      Loftus, G.R., & Masson, M.E. (1994). Using confidence intervals in
          within-subjects designs. The Psychonomic Bulletin & Review, 1(4),
          476-490.
      Masson, M.E., & Loftus, G.R. (2003). Using confidence intervals for
          graphically-based data interpretation. Canadian Journal of
          Experimental Pscyhology, 57(3), 203-220.
      Muller, K.E. & Barton, C. N. (1989). Approximating power for repeated-
          measures ANOVA lacking sphericity. Journal of the American
          Statistical Association, 84 (406), 549-555.
    """
    
    results_obj = AnovaResults()
    if wfactors == None:
        wfactors = []
        
    if bfactors == None:
        bfactors = []

    ## Intialize results_obj variables
    
    # holds a reference to a pandas.DataFrame object
    results_obj.df=dataframe

    # a string label/key to the dependent variable data in results_obj.df
    results_obj.dv=dv

    # a list of within-participant factors
    results_obj.wfactors=wfactors
    
    # a list of between-participant factors
    results_obj.bfactors=bfactors

    # a string label/key to the participant/case data in results_obj.df
    results_obj.sub=sub

    results_obj.plots=[]

    # a list of all the factors
    factors=wfactors+bfactors
    results_obj.dftrim=0.

    # check to see if a data should be transformed
    if   transform in ['log','log10']:
        results_obj.transform=np.log10
        tstr='LOG_'
        
    elif transform in ['reciprocal','inverse']:
        results_obj.transform=lambda X:1./np.array(X)
        tstr='RECIPROCAL_'
        
    elif transform in ['square-root','sqrt']:
        results_obj.transform=np.sqrt
        tstr='SQRT_'
        
    elif transform in ['arcsine','arcsin'] :
        results_obj.transform=np.arcsin
        tstr='ARCSIN_'
        
    elif transform in ['windsor01']:
        results_obj.transform=lambda X: windsor(np.array(X),.01)
        tstr='WINDSOR01_'
        
    elif transform in ['windsor05']:
        results_obj.transform=lambda X: windsor(np.array(X),.05)
        tstr='WINDSOR05_'
        
    elif transform in ['windsor10']:
        results_obj.transform=lambda X: windsor(np.array(X),.10)
        tstr='WINDSOR10_'        

    if transform!='':
        if 'windsor' in transform:
            results_obj.df[tstr + results_obj.dv], \
            results_obj.dftrim = results_obj.transform(results_obj.df[results_obj.dv])
##                print('%i degrees of freedom lost from trim'%int(results_obj.dftrim))
        else:
            results_obj.df[tstr+results_obj.dv] = results_obj.transform(results_obj.df[results_obj.dv])
            
        results_obj.dv = tstr+results_obj.dv
        
    # results_obj.pt is a PyvtTbl (list of lists)
    #     rows = replications (eg subjects)
    #     columns = conditions
    results_obj.pt = pandas.pivot_table(results_obj.df, values=results_obj.dv,
                                 rows=[results_obj.sub],cols=factors)

    # results_obj.pt_asarray is the same data as results_obj.pt except as a numpy array
    results_obj.pt_asarray = np.array(results_obj.pt, dtype=np.float64)

    # Replace NaN values with mean of dv
    results_obj.pt_asarray[np.isnan(results_obj.pt_asarray)] = np.mean(results_obj.df[results_obj.dv])
    
    # A vector with as many entries as factors, each entry being
    # the number of levels for that factor
    #
    # Data matrix results_obj.pt_asarray must have as many columns (conditions)
    # as the product of the elements of the factor matrix D
    #
    # First factor rotates slowest; last factor fastest
    results_obj.D=[len(set(results_obj.df[f])) for f in factors]
    results_obj.numlevels = dict([(f, results_obj.D[i]) for i,f in enumerate(factors)])
    results_obj.numlevels[sub] = len(set(results_obj.df[sub]))

    if len(wfactors)!=0 and len(bfactors)==0:
        _within(results_obj)
        
    if len(wfactors)==0 and len(bfactors)!=0:
        _between(results_obj)
        
    if len(wfactors)!=0 and len(bfactors)!=0:
        _mixed(results_obj)
        
    return results_obj
