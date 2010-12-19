'''

from pystatsmodels mailinglist 20100524

Notes:
 - unfinished, unverified, but most parts seem to work in MonteCarlo
 - one example taken from lecture notes looks ok
 - needs cases with non-monotonic inequality for test to see difference between
   one-step, step-up and step-down procedures
 - FDR doesn't look really better then Bonferoni in the MC examples that I tried
update:
 - now tested against R, stats and multtest, have all methods
 - getting Hommel was impossible until I found reference for pvalue correction
 - now, since I have p-values correction, some of the original tests
   implementation is not really needed anymore. I think I keep it for reference.
   Test procedure for Hommel in development session log
 - I haven't updated other functions and classes in here.
   - multtest has some good helper function according to docs
 - still need to update references, the real papers
 - fdr with estimated true hypothesis still missing


some References:

Gibbons, Jean Dickinson and Chakraborti Subhabrata, 2003, Nonparametric Statistical
Inference, Fourth Edition, Marcel Dekker
    p.363: 10.4 THE KRUSKAL-WALLIS ONE-WAY ANOVA TEST AND MULTIPLE COMPARISONS
    p.367: multiple comparison for kruskal formula used in multicomp.kruskal

Sheskin, David J., 2004, Handbook of Parametric and Nonparametric Statistical
Procedures, 3rd ed., Chapman&Hall/CRC
    Test 21: The Single-Factor Between-Subjects Analysis of Variance
    Test 22: The Kruskal-Wallis One-Way Analysis of Variance by Ranks Test

Zwillinger, Daniel and Stephen Kokoska, 2000, CRC standard probability and
statistics tables and formulae, Chapman&Hall/CRC
    14.9 WILCOXON RANKSUM (MANN WHITNEY) TEST

Author: Josef Pktd and example from H Raja and rewrite from Vincent Davis


TODO

handle exception if empty, shows up only sometimes when running this
- DONE I think

Traceback (most recent call last):
  File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\stats\multicomp.py", line 711, in <module>
    print 'sh', multipletests(tpval, alpha=0.05, method='sh')
  File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\stats\multicomp.py", line 241, in multipletests
    rejectmax = np.max(np.nonzero(reject))
  File "C:\Programs\Python25\lib\site-packages\numpy\core\fromnumeric.py", line 1765, in amax
    return _wrapit(a, 'max', axis, out)
  File "C:\Programs\Python25\lib\site-packages\numpy\core\fromnumeric.py", line 37, in _wrapit
    result = getattr(asarray(obj),method)(*args, **kwds)
ValueError: zero-size array to ufunc.reduce without identity


'''



import xlrd
import xlwt
import scipy.stats
import numpy
import numpy as np
import math
from scipy import stats
import scikits.statsmodels as sm
from numpy.testing import assert_almost_equal


qcrit = '''
	2 		3 		4 		5 		6 		7 		8 		9 		10
5 	3.64 5.70 	4.60 6.98 	5.22 7.80 	5.67 8.42 	6.03 8.91 	6.33 9.32 	6.58 9.67 	6.80 9.97 	6.99 10.24
6 	3.46 5.24 	4.34 6.33 	4.90 7.03 	5.30 7.56 	5.63 7.97 	5.90 8.32 	6.12 8.61 	6.32 8.87 	6.49 9.10
7 	3.34 4.95 	4.16 5.92 	4.68 6.54 	5.06 7.01 	5.36 7.37 	5.61 7.68 	5.82 7.94 	6.00 8.17 	6.16 8.37
8 	3.26 4.75 	4.04 5.64 	4.53 6.20 	4.89 6.62 	5.17 6.96 	5.40 7.24       5.60 7.47 	5.77 7.68 	5.92 7.86
9 	3.20 4.60 	3.95 5.43 	4.41 5.96 	4.76 6.35 	5.02 6.66 	5.24 6.91       5.43 7.13 	5.59 7.33 	5.74 7.49
10 	3.15 4.48 	3.88 5.27 	4.33 5.77 	4.65 6.14 	4.91 6.43 	5.12 6.67       5.30 6.87 	5.46 7.05 	5.60 7.21
11 	3.11 4.39 	3.82 5.15 	4.26 5.62 	4.57 5.97 	4.82 6.25 	5.03 6.48	5.20 6.67 	5.35 6.84 	5.49 6.99
12 	3.08 4.32 	3.77 5.05 	4.20 5.50 	4.51 5.84 	4.75 6.10 	4.95 6.32	5.12 6.51 	5.27 6.67 	5.39 6.81
13 	3.06 4.26 	3.73 4.96 	4.15 5.40 	4.45 5.73 	4.69 5.98 	4.88 6.19	5.05 6.37 	5.19 6.53 	5.32 6.67
14 	3.03 4.21 	3.70 4.89 	4.11 5.32 	4.41 5.63 	4.64 5.88 	4.83 6.08	4.99 6.26 	5.13 6.41 	5.25 6.54
15 	3.01 4.17 	3.67 4.84 	4.08 5.25 	4.37 5.56 	4.59 5.80 	4.78 5.99	4.94 6.16 	5.08 6.31 	5.20 6.44
16 	3.00 4.13 	3.65 4.79 	4.05 5.19 	4.33 5.49 	4.56 5.72 	4.74 5.92	4.90 6.08 	5.03 6.22 	5.15 6.35
17 	2.98 4.10 	3.63 4.74 	4.02 5.14 	4.30 5.43 	4.52 5.66 	4.70 5.85	4.86 6.01 	4.99 6.15 	5.11 6.27
18 	2.97 4.07 	3.61 4.70 	4.00 5.09 	4.28 5.38 	4.49 5.60 	4.67 5.79	4.82 5.94 	4.96 6.08 	5.07 6.20
19 	2.96 4.05 	3.59 4.67 	3.98 5.05 	4.25 5.33 	4.47 5.55 	4.65 5.73	4.79 5.89 	4.92 6.02 	5.04 6.14
20 	2.95 4.02 	3.58 4.64 	3.96 5.02 	4.23 5.29 	4.45 5.51 	4.62 5.69	4.77 5.84 	4.90 5.97 	5.01 6.09
24 	2.92 3.96 	3.53 4.55 	3.90 4.91 	4.17 5.17 	4.37 5.37 	4.54 5.54	4.68 5.69 	4.81 5.81 	4.92 5.92
30 	2.89 3.89 	3.49 4.45 	3.85 4.80 	4.10 5.05 	4.30 5.24 	4.46 5.40	4.60 5.54 	4.72 5.65 	4.82 5.76
40 	2.86 3.82 	3.44 4.37 	3.79 4.70 	4.04 4.93 	4.23 5.11 	4.39 5.26	4.52 5.39 	4.63 5.50 	4.73 5.60
60 	2.83 3.76 	3.40 4.28 	3.74 4.59 	3.98 4.82 	4.16 4.99 	4.31 5.13	4.44 5.25 	4.55 5.36 	4.65 5.45
120 	2.80 3.70 	3.36 4.20 	3.68 4.50 	3.92 4.71 	4.10 4.87 	4.24 5.01	4.36 5.12 	4.47 5.21 	4.56 5.30
infinity 	2.77 3.64 	3.31 4.12 	3.63 4.40 	3.86 4.60 	4.03 4.76 	4.17 4.88 	4.29 4.99 	4.39 5.08 	4.47 5.16
'''

res = [line.split() for line in qcrit.replace('infinity','9999').split('\n')]
c=np.array(res[2:-1]).astype(float)
#c[c==9999] = np.inf
ccols = np.arange(2,11)
crows = c[:,0]
cv005 = c[:, 1::2]
cv001 = c[:, 2::2]

from scipy import interpolate
def get_tukeyQcrit(k, df, alpha=0.05):
    '''
    return critical values for Tukey's HSD (Q)

    Parameters
    ----------
    k : int in {2, ..., 10}
        number of tests
    df : int
        degrees of freedom of error term
    alpha : {0.05, 0.01}
        type 1 error, 1-confidence level



    not enough error checking for limitations
    '''
    if alpha == 0.05:
        intp = interpolate.interp1d(crows, cv005[:,k-2])
    elif alpha == 0.01:
        intp = interpolate.interp1d(crows, cv001[:,k-2])
    return intp(df)


def Tukeythreegene(first,second,third):
    #Performing the Tukey HSD post-hoc test for three genes
##   qwb = xlrd.open_workbook('F:/Lab/bioinformatics/qcrittable.xls')
##    #opening the workbook containing the q crit table
##   qwb.sheet_names()
##   qcrittable = qwb.sheet_by_name(u'Sheet1')

   firstmean = numpy.mean(first) #means of the three arrays
   secondmean = numpy.mean(second)
   thirdmean = numpy.mean(third)

   firststd = numpy.std(first) #standard deviations of the threearrays
   secondstd = numpy.std(second)
   thirdstd = numpy.std(third)

   firsts2 = math.pow(firststd,2) #standard deviation squared of the three arrays
   seconds2 = math.pow(secondstd,2)
   thirds2 = math.pow(thirdstd,2)

   mserrornum = firsts2*2+seconds2*2+thirds2*2 #numerator for mean square error
   mserrorden = (len(first)+len(second)+len(third))-3 #denominator for mean square error
   mserror = mserrornum/mserrorden #mean square error

   standarderror = math.sqrt(mserror/len(first))
   #standard error, which is square root of mserror and the number of samples in a group

   dftotal = len(first)+len(second)+len(third)-1 #various degrees of freedom
   dfgroups = 2
   dferror = dftotal-dfgroups

   qcrit = 0.5 # fix arbitrary#qcrittable.cell(dftotal, 3).value
   qcrit = get_tukeyQcrit(3, dftotal, alpha=0.05)
   #getting the q critical value, for degrees of freedom total and 3 groups

   qtest3to1 = (math.fabs(thirdmean-firstmean))/standarderror
    #calculating q test statistic values
   qtest3to2 = (math.fabs(thirdmean-secondmean))/standarderror
   qtest2to1 = (math.fabs(secondmean-firstmean))/standarderror

   conclusion = []

##    print qcrit
   print qtest3to1
   print qtest3to2
   print qtest2to1

   if(qtest3to1>qcrit): #testing all q test statistic values to q critical values
       conclusion.append('3to1null')
   else:
       conclusion.append('3to1alt')
   if(qtest3to2>qcrit):
       conclusion.append('3to2null')
   else:
       conclusion.append('3to2alt')
   if(qtest2to1>qcrit):
       conclusion.append('2to1null')
   else:
       conclusion.append('2to1alt')

   return conclusion


#rewrite by Vincent
def Tukeythreegene2(genes): #Performing the Tukey HSD post-hoc test for three genes
   """gend is a list, ie [first, second, third]"""
   qwb = xlrd.open_workbook('F:/Lab/bioinformatics/qcrittable.xls')
    #opening the workbook containing the q crit table
   qwb.sheet_names()
   qcrittable = qwb.sheet_by_name(u'Sheet1')

   means = []
   stds = []
   for gene in genes:
      means.append(numpy.mean(gene))
      std.append(numpy.std(gene))

   #firstmean = numpy.mean(first) #means of the three arrays
   #secondmean = numpy.mean(second)
   #thirdmean = numpy.mean(third)

   #firststd = numpy.std(first) #standard deviations of the three arrays
   #secondstd = numpy.std(second)
   #thirdstd = numpy.std(third)

   stds2 = []
   for std in stds:
      stds2.append(math.pow(std,2))


   #firsts2 = math.pow(firststd,2) #standard deviation squared of the three arrays
   #seconds2 = math.pow(secondstd,2)
   #thirds2 = math.pow(thirdstd,2)

   #mserrornum = firsts2*2+seconds2*2+thirds2*2 #numerator for mean square error
   mserrornum = sum(stds2)*2
   mserrorden = (len(genes[0])+len(genes[1])+len(genes[2]))-3 #denominator for mean square error
   mserror = mserrornum/mserrorden #mean square error


def catstack(args):
    x = np.hstack(args)
    labels = np.hstack([k*np.ones(len(arr)) for k,arr in enumerate(args)])
    return x, labels


def multipletests(pvals, alpha=0.05, method='hs', returnsorted=False):
    '''test results and p-value correction for multiple tests


    Parameters
    ----------
    pvals : array_like
        uncorrected p-values
    alpha : float
        FWER, family-wise error rate, e.g. 0.1
    method : string
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are ::

        `bonferroni` : one-step correction
        `sidak` : on-step correction
        `holm-sidak` :
        `holm` :
        `simes-hochberg` :
        `hommel` :
        `fdr_bh` : Benjamini/Hochberg
        `fdr_by` : Benjamini/Yekutieli
    returnsorted : bool
         not tested, return sorted p-values instead of original sequence

    Returns
    -------
    reject : array, boolean
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : array
        p-values corrected for multiple tests
    alphacSidak: float
        corrected pvalue with Sidak method
    alphacBonf: float
        corrected pvalue with Sidak method


    Notes
    -----
    all corrected pvalues now tested against R.
    insufficient "cosmetic" tests yet

    there will be API changes.

    References
    ----------

    '''
    pvals = np.asarray(pvals)
    alphaf = alpha  # Notation ?
    sortind = np.argsort(pvals)
    pvals = pvals[sortind]
    sortrevind = sortind.argsort()
    ntests = len(pvals)
    alphacSidak = 1 - np.power((1. - alphaf), 1./ntests)
    alphacBonf = alphaf / float(ntests)
    if method.lower() in ['b', 'bonf', 'bonferroni']:
        reject = pvals < alphacBonf
        pvals_corrected = pvals * float(ntests)  # not sure

    elif method.lower() in ['s', 'sidak']:
        reject = pvals < alphacSidak
        pvals_corrected = 1 - np.power((1. - pvals), ntests)  # not sure

    elif method.lower() in ['hs', 'holm-sidak']:
        notreject = pvals > alphacSidak
        notrejectmin = np.min(np.nonzero(notreject))
        notreject[notrejectmin:] = True
        reject = ~notreject
        pvals_corrected = None  # not yet implemented
        #TODO: new not tested, mainly guessing by analogy
        pvals_corrected_raw = 1 - np.power((1. - pvals), np.arange(ntests, 0, -1))#ntests) # from "sidak" #pvals / alphacSidak * alphaf
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)

    elif method.lower() in ['h', 'holm']:
        notreject = pvals > alphaf / np.arange(ntests, 0, -1) #alphacSidak
        notrejectmin = np.min(np.nonzero(notreject))
        notreject[notrejectmin:] = True
        reject = ~notreject
        pvals_corrected = None  # not yet implemented
        #TODO: new not tested, mainly guessing by analogy
        pvals_corrected_raw = pvals * np.arange(ntests, 0, -1) #ntests) # from "sidak" #pvals / alphacSidak * alphaf
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)

    elif method.lower() in ['sh', 'simes-hochberg']:
        alphash = alphaf / np.arange(ntests, 0, -1)
        reject = pvals < alphash
        rejind = np.nonzero(reject)
        if rejind[0].size > 0:
            rejectmax = np.max(np.nonzero(reject))
            reject[:rejectmax] = True
        #check else
        pvals_corrected = None  # not yet implemented
        #TODO: new not tested, mainly guessing by analogy, looks ok in 1 example
        pvals_corrected_raw = np.arange(ntests, 0, -1) * pvals
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]

    elif method.lower() in ['ho', 'hommel']:
        a=pvals.copy()
        for m in range(ntests, 1, -1):
            cim = np.min(m * pvals[-m:] / np.arange(1,m+1.))
            a[-m:] = np.maximum(a[-m:], cim)
            a[:-m] = np.maximum(a[:-m], np.minimum(m * pvals[:-m], cim))
        pvals_corrected = a
        reject = a < alphaf

    elif method.lower() in ['fdr_bh', 'fdr_i', 'fdr_p', 'fdri', 'fdrp']:
        #delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection0(pvals, alpha=alpha,
                                                 method='indep')
    elif method.lower() in ['fdr_by', 'fdr_n', 'fdr_c', 'fdrn', 'fdrcorr']:
        #delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection0(pvals, alpha=alpha,
                                                 method='n')
    else:
        raise ValueError('method not recognized')


    if not pvals_corrected is None:
        pvals_corrected[pvals_corrected>1] = 1
    if returnsorted:
        return reject, pvals_corrected, alphacSidak, alphacBonf
    else:
        if pvals_corrected is None:
            return reject[sortrevind], pvals_corrected, alphacSidak, alphacBonf
        else:
            return reject[sortrevind], pvals_corrected[sortrevind], alphacSidak, alphacBonf



def maxzero(x):
    '''find all up zero crossings and return the index of the highest


    >>> np.random.seed(12345)
    >>> x = np.random.randn(8)
    >>> x
    array([-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057,
            1.39340583,  0.09290788,  0.28174615])
    >>> maxzero(x)
    (4, array([1, 4]))


    no up-zero-crossing at end

    >>> np.random.seed(0)
    >>> x = np.random.randn(8)
    >>> x
    array([ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
           -0.97727788,  0.95008842, -0.15135721])
    >>> maxzero(x)
    (None, array([6]))
'''
    x = np.asarray(x)
    cond1 = x[:-1] < 0
    cond2 = x[1:] > 0
    #allzeros = np.nonzero(np.sign(x[:-1])*np.sign(x[1:]) <= 0)[0] + 1
    allzeros = np.nonzero((cond1 & cond2) | (x[1:]==0))[0] + 1
    if x[-1] >=0:
        maxz = max(allzeros)
    else:
        maxz = None
    return maxz, allzeros

def maxzerodown(x):
    '''find all up zero crossings and return the index of the highest


    >>> np.random.seed(12345)
    >>> x = np.random.randn(8)
    >>> x
    array([-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057,
            1.39340583,  0.09290788,  0.28174615])
    >>> maxzero(x)
    (4, array([1, 4]))


    no up-zero-crossing at end

    >>> np.random.seed(0)
    >>> x = np.random.randn(8)
    >>> x
    array([ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
           -0.97727788,  0.95008842, -0.15135721])
    >>> maxzero(x)
    (None, array([6]))
'''
    x = np.asarray(x)
    cond1 = x[:-1] > 0
    cond2 = x[1:] < 0
    #allzeros = np.nonzero(np.sign(x[:-1])*np.sign(x[1:]) <= 0)[0] + 1
    allzeros = np.nonzero((cond1 & cond2) | (x[1:]==0))[0] + 1
    if x[-1] <=0:
        maxz = max(allzeros)
    else:
        maxz = None
    return maxz, allzeros

def ecdf(x):
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

def rejectionline(n, alpha=0.5):
    '''

    from: section 3.2, page 60
    '''
    t = np.arange(n)/float(n)
    frej = t/( t * (1-alpha) + alpha)
    return frej

def fdrcorrection0(pvals, alpha=0.05, method='indep'):
    '''
    '''
    pvals = np.asarray(pvals)

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = ecdf(pvals_sorted) / cm
##    elif method in ['n', 'negcorr']:
##        cm = np.sum(np.arange(len(pvals)))
##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and necorr implemented')
    reject = pvals_sorted < ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected>1] = 1
    return reject[sortrevind], pvals_corrected[sortrevind]
    #return reject[pvals_sortind.argsort()]


#I don't remember what I changed or why 2 versions,
#this follows german diss ???  with rline
#this might be useful if the null hypothesis is not "all effects are zero"
#rename to _bak and working again on fdrcorrection0
def fdrcorrection_bak(pvals, alpha=0.05, method='indep'):
    '''
    Reject False discovery rate correction for pvalues


    missing: methods that estimate fraction of true hypotheses

    '''
    pvals = np.asarray(pvals)


    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    pecdf = ecdf(pvals_sorted)
    if method in ['i', 'indep', 'p', 'poscorr']:
        rline = pvals_sorted / alpha
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals)))
        rline = pvals_sorted / alpha * cm
    elif method in ['g', 'onegcorr']:  #what's this ? german diss
        rline = pvals_sorted / (pvals_sorted*(1-alpha) + alpha)
    elif method in ['oth', 'o2negcorr']: # other invalid, cut-paste
        cm = np.sum(np.arange(len(pvals)))
        rline = pvals_sorted / alpha /cm
    else:
        raise ValueError('method not available')

    reject = pecdf >= rline
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True
    return reject[pvals_sortind.argsort()]

def mcfdr(nrepl=100, nobs=50, ntests=10, ntrue=6, mu=0.5, alpha=0.05, rho=0.):
    nfalse = ntests - ntrue
    locs = np.array([0.]*ntrue + [mu]*(ntests - ntrue))
    results = []
    for i in xrange(nrepl):
        #rvs = locs + stats.norm.rvs(size=(nobs, ntests))
        rvs = locs + randmvn(rho, size=(nobs, ntests))
        tt, tpval = stats.ttest_1samp(rvs, 0)
        res = fdrcorrection_bak(np.abs(tpval), alpha=alpha, method='i')
        res0 = fdrcorrection0(np.abs(tpval), alpha=alpha)
        #res and res0 give the same results
        results.append([np.sum(res[:ntrue]), np.sum(res[ntrue:])] +
                       [np.sum(res0[:ntrue]), np.sum(res0[ntrue:])] +
                       res.tolist() +
                       np.sort(tpval).tolist() +
                       [np.sum(tpval[:ntrue]<alpha),
                        np.sum(tpval[ntrue:]<alpha)] +
                       [np.sum(tpval[:ntrue]<alpha/ntests),
                        np.sum(tpval[ntrue:]<alpha/ntests)])
    return np.array(results)

def randmvn(rho, size=(1, 2), standardize=False):
    nobs, nvars = size
    if 0 < rho and rho < 1:
        rvs = np.random.randn(nobs, nvars+1)
        rvs2 = rvs[:,:-1] * np.sqrt((1-rho)) + rvs[:,-1:] * np.sqrt(rho)
    elif rho ==0:
        rvs2 = np.random.randn(nobs, nvars)
    elif rho < 0:
        if rho < -1./(nvars-1):
            raise ValueError('rho has to be larger than -1./(nvars-1)')
        elif rho == -1./(nvars-1):
            rho = -1./(nvars-1+1e-10)  #barely positive definite
        #use Cholesky
        A = rho*np.ones((nvars,nvars))+(1-rho)*np.eye(nvars)
        rvs2 = np.dot(np.random.randn(nobs, nvars), np.linalg.cholesky(A).T)
    if standardize:
        rvs2 = stats.zscore(rvs2)
    return rvs2

##############

def tiecorrect(xranks):
    '''

    should be equivalent of scipy.stats.tiecorrect

    '''
    #casting to int rounds down, but not relevant for this case
    rankbincount = np.bincount(np.asarray(xranks,dtype=int))
    nties = rankbincount[rankbincount > 1]
    ntot = float(len(xranks));
    tiecorrection = 1 - (nties**3 - nties).sum()/(ntot**3 - ntot)
    return tiecorrection


class GroupsStats(object):
    '''
    statistics by groups (another version)

    groupstats as a class with lazy evaluation (not yet - decorators are still
    missing)

    written this time as equivalent of scipy.stats.rankdata
    gs = GroupsStats(X, useranks=True)
    assert_almost_equal(gs.groupmeanfilter, stats.rankdata(X[:,0]), 15)

    '''

    def __init__(self, x, useranks=False, uni=None, intlab=None):
        '''

        Parameters
        ----------
        x : array, 2d
            first column data, second column group labels
        useranks : boolean
            if true, then use ranks as data corresponding to the
            scipy.stats.rankdata definition (start at 1, ties get mean)
        uni, intlab : arrays (optional)
            to avoid call to unique, these can be given as inputs


        '''
        self.x = np.asarray(x)
        if intlab is None:
            uni, intlab = np.unique(x[:,1], return_inverse=True)
        elif uni is None:
            uni = np.unique(x[:,1])

        self.useranks = useranks


        self.uni = uni
        self.intlab = intlab
        self.groupnobs = groupnobs = np.bincount(intlab)

        #temporary until separated and made all lazy
        self.runbasic(useranks=useranks)



    def runbasic_old(self, useranks=False):
        #check: refactoring screwed up case useranks=True

        #groupxsum = np.bincount(intlab, weights=X[:,0])
        #groupxmean = groupxsum * 1.0 / groupnobs
        x = self.x
        if useranks:
            self.xx = x[:,1].argsort().argsort() + 1  #rankraw
        else:
            self.xx = x[:,0]
        self.groupsum = groupranksum = np.bincount(self.intlab, weights=self.xx)
        #print 'groupranksum', groupranksum, groupranksum.shape, self.groupnobs.shape
        # start at 1 for stats.rankdata :
        self.groupmean = grouprankmean = groupranksum * 1.0 / self.groupnobs # + 1
        self.groupmeanfilter = grouprankmean[self.intlab]
        #return grouprankmean[intlab]

    def runbasic(self, useranks=False):
        #check: refactoring screwed up case useranks=True

        #groupxsum = np.bincount(intlab, weights=X[:,0])
        #groupxmean = groupxsum * 1.0 / groupnobs
        x = self.x
        if useranks:
            xuni, xintlab = np.unique(x[:,0], return_inverse=True)
            ranksraw = x[:,0].argsort().argsort() + 1  #rankraw
            self.xx = GroupsStats(np.column_stack([ranksraw, xintlab]),
                                  useranks=False).groupmeanfilter
        else:
            self.xx = x[:,0]
        self.groupsum = groupranksum = np.bincount(self.intlab, weights=self.xx)
        #print 'groupranksum', groupranksum, groupranksum.shape, self.groupnobs.shape
        # start at 1 for stats.rankdata :
        self.groupmean = grouprankmean = groupranksum * 1.0 / self.groupnobs # + 1
        self.groupmeanfilter = grouprankmean[self.intlab]
        #return grouprankmean[intlab]

    def groupdemean(self):
        return self.xx - self.groupmeanfilter

    def groupsswithin(self):
        xtmp = self.groupdemean()
        return np.bincount(self.intlab, weights=xtmp**2)

    def groupvarwithin(self):
        return self.groupsswithin()/(self.groupnobs-1).sum()


class MultiComparison(object):
    '''Tests for multiple comparisons


    '''

    def __init__(self, x, groups):
        self.data = x
        self.groups = groups
        self.groupsunique, self.groupintlab = np.unique(groups, return_inverse=True)
        self.datali = [x[groups == k] for k in self.groupsunique]
        self.pairindices = np.triu_indices(len(self.groupsunique),1)  #tuple
        self.nobs = x.shape[0]

    def getranks(self):
        #bug: the next should use self.groupintlab instead of self.groups
        #self.ranks = GroupsStats(np.column_stack([self.data, self.groups]),
        self.ranks = GroupsStats(np.column_stack([self.data, self.groupintlab]),
                                 useranks=True)
        self.rankdata = self.ranks.groupmeanfilter



    def kruskal(self, pairs=None, multimethod='T'):
        '''
        pairwise comparison for kruskal-wallis test

        '''
        self.getranks()
        tot = self.nobs
        meanranks = self.ranks.groupmean
        groupnobs = self.ranks.groupnobs


        # simultaneous/separate treatment of multiple tests
        f=(tot*(tot+1.)/12.)/stats.tiecorrect(self.rankdata) #(xranks)
        print 'MultiComparison.kruskal'
        for i,j in zip(*self.pairindices):
            #pdiff = np.abs(mrs[i] - mrs[j])
            pdiff = np.abs(meanranks[i] - meanranks[j])
            se = np.sqrt(f * np.sum(1./groupnobs[[i,j]] )) #np.array([8,8]))) #Fixme groupnobs[[i,j]] ))
            Q = pdiff/se

            print i,j, pdiff, se, pdiff/se, pdiff/se>2.6310,
            print stats.norm.sf(Q)*2
            return stats.norm.sf(Q)*2


    def allpairtest(self, testfunc, alpha=0.05, method='bonf'):
        '''

        errors:
        results from multipletests are in different order
        pval_corrected can be larger than 1 ???
        '''
        res = []
        for i,j in zip(*self.pairindices):
            res.append(testfunc(self.datali[i], self.datali[j]))
        res = np.array(res)
        reject, pvals_corrected, alphacSidak, alphacBonf = \
                multipletests(res[:,1], alpha=0.05, method=method)
        #print np.column_stack([res[:,0],res[:,1], reject, pvals_corrected])

        i1, i2 = self.pairindices
        if pvals_corrected is None:
            resarr = np.array(zip(i1, i2,
                                  np.round(res[:,0],4),
                                  np.round(res[:,1],4),
                                  reject),
                       dtype=[('group1', int),
                              ('group2', int),
                              ('stat',float),
                              ('pval',float),
                              ('reject', np.bool8)])
        else:
            resarr = np.array(zip(i1, i2,
                                  np.round(res[:,0],4),
                                  np.round(res[:,1],4),
                                  np.round(pvals_corrected,4),
                                  reject),
                       dtype=[('group1', int),
                              ('group2', int),
                              ('stat',float),
                              ('pval',float),
                              ('pval_corr',float),
                              ('reject', np.bool8)])
        summtab = sm.iolib.SimpleTable(resarr, headers=resarr.dtype.names)
        summtab.title = 'Test Multiple Comparison %s \n%s%4.2f method=%s' % (testfunc.__name__,
                        'FWER=', alpha, method) + \
                        '\nalphacSidak=%4.2f, alphacBonf=%5.3f' % (alphacSidak, alphacBonf)
        return summtab, (res, reject, pvals_corrected, alphacSidak, alphacBonf), resarr










def rankdata(x):
    uni, intlab = np.unique(x[:,0], return_inverse=True)
    groupnobs = np.bincount(intlab)
    groupxsum = np.bincount(intlab, weights=X[:,0])
    groupxmean = groupxsum * 1.0 / groupnobs

    rankraw = x[:,0].argsort().argsort()
    groupranksum = np.bincount(intlab, weights=rankraw)
    # start at 1 for stats.rankdata :
    grouprankmean = groupranksum * 1.0 / groupnobs + 1
    return grouprankmean[intlab]

if __name__ == '__main__':

    examples = []#['tukey', 'tukeycrit', 'fdr', 'fdrmc', 'bonf', 'randmvn',
                #'multicompdev'][2:3]

    if 'tukey' in examples:
        #Example Tukey
        x = np.array([[0,0,1]]).T + np.random.randn(3, 20)
        print Tukeythreegene(*x)

        #Example FDR
        #------------

    if ('fdr' in examples) or ('bonf' in examples):
        x1 = [1,1,1,0,-1,-1,-1,0,1,1,-1,1]
        print zip(np.arange(len(x1)), x1)
        print maxzero(x1)
        #[(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -1), (6, -1), (7, 0), (8, 1), (9, 1), (10, -1), (11, 1)]
        #(11, array([ 3,  7, 11]))

        print maxzerodown(-np.array(x1))

        locs = np.linspace(0,1,10)
        locs = np.array([0.]*6 + [0.75]*4)
        rvs = locs + stats.norm.rvs(size=(20,10))
        tt, tpval = stats.ttest_1samp(rvs, 0)
        tpval_sortind = np.argsort(tpval)
        tpval_sorted = tpval[tpval_sortind]

        reject = tpval_sorted < ecdf(tpval_sorted)*0.05
        reject2 = max(np.nonzero(reject))
        print reject

        res = np.array(zip(np.round(rvs.mean(0),4),np.round(tpval,4),
                           reject[tpval_sortind.argsort()]),
                       dtype=[('mean',float),
                              ('pval',float),
                              ('reject', np.bool8)])
        print sm.iolib.SimpleTable(res, headers=res.dtype.names)
        print fdrcorrection_bak(tpval, alpha=0.05)
        print reject

        print '\nrandom example'
        print 'bonf', multipletests(tpval, alpha=0.05, method='bonf')
        print 'sidak', multipletests(tpval, alpha=0.05, method='sidak')
        print 'hs', multipletests(tpval, alpha=0.05, method='hs')
        print 'sh', multipletests(tpval, alpha=0.05, method='sh')
        pvals = np.array('0.0020 0.0045 0.0060 0.0080 0.0085 0.0090 0.0175 0.0250 '
                 '0.1055 0.5350'.split(), float)
        print '\nexample from lecturnotes'
        for meth in ['bonf', 'sidak', 'hs', 'sh']:
            print meth, multipletests(pvals, alpha=0.05, method=meth)

    if 'fdrmc' in examples:
        mcres = mcfdr(nobs=100, nrepl=1000, ntests=30, ntrue=30, mu=0.1, alpha=0.05, rho=0.3)
        mcmeans = np.array(mcres).mean(0)
        print mcmeans
        print mcmeans[0]/6., 1-mcmeans[1]/4.
        print mcmeans[:4], mcmeans[-4:]


    if 'randmvn' in examples:
        rvsmvn = randmvn(0.8, (5000,5))
        print np.corrcoef(rvsmvn, rowvar=0)
        print rvsmvn.var(0)


    if 'tukeycrit' in examples:
        print get_tukeyQcrit(8, 8, alpha=0.05), 5.60
        print get_tukeyQcrit(8, 8, alpha=0.01), 7.47


    if 'multicompdev' in examples:
        #development of kruskal-wallis multiple-comparison
        #example from matlab file exchange

        X = np.array([[7.68, 1], [7.69, 1], [7.70, 1], [7.70, 1], [7.72, 1],
                      [7.73, 1], [7.73, 1], [7.76, 1], [7.71, 2], [7.73, 2],
                      [7.74, 2], [7.74, 2], [7.78, 2], [7.78, 2], [7.80, 2],
                      [7.81, 2], [7.74, 3], [7.75, 3], [7.77, 3], [7.78, 3],
                      [7.80, 3], [7.81, 3], [7.84, 3], [7.71, 4], [7.71, 4],
                      [7.74, 4], [7.79, 4], [7.81, 4], [7.85, 4], [7.87, 4],
                      [7.91, 4]])
        xli = [X[X[:,1]==k,0] for k in range(1,5)]
        xranks = stats.rankdata(X[:,0])
        xranksli = [xranks[X[:,1]==k] for k in range(1,5)]
        xnobs = np.array([len(x) for x in xli])
        meanranks = [item.mean() for item in xranksli]
        sumranks = [item.sum() for item in xranksli]
        # equivalent function
        #from scipy import special
        #-np.sqrt(2.)*special.erfcinv(2-0.5) == stats.norm.isf(0.25)
        stats.norm.sf(0.67448975019608171)
        stats.norm.isf(0.25)

        mrs = np.sort(meanranks)
        v1, v2 = np.triu_indices(4,1)
        print mrs[v2] - mrs[v1]
        diffidx = np.argsort(mrs[v2] - mrs[v1])[::-1]
        mrs[v2[diffidx]] - mrs[v1[diffidx]]

        print 'kruskal for all pairs'
        for i,j in zip(v2[diffidx], v1[diffidx]):
            print i,j, stats.kruskal(xli[i], xli[j]),
            mwu, mwupval = stats.mannwhitneyu(xli[i], xli[j], use_continuity=False)
            print mwu, mwupval*2, mwupval*2<0.05/6., mwupval*2<0.1/6.





        uni, intlab = np.unique(X[:,0], return_inverse=True)
        groupnobs = np.bincount(intlab)
        groupxsum = np.bincount(intlab, weights=X[:,0])
        groupxmean = groupxsum * 1.0 / groupnobs

        rankraw = X[:,0].argsort().argsort()
        groupranksum = np.bincount(intlab, weights=rankraw)
        # start at 1 for stats.rankdata :
        grouprankmean = groupranksum * 1.0 / groupnobs + 1
        assert_almost_equal(grouprankmean[intlab], stats.rankdata(X[:,0]), 15)
        gs = GroupsStats(X, useranks=True)
        print gs.groupmeanfilter
        print grouprankmean[intlab]
        #the following has changed
        #assert_almost_equal(gs.groupmeanfilter, stats.rankdata(X[:,0]), 15)

        xuni, xintlab = np.unique(X[:,0], return_inverse=True)
        gs2 = GroupsStats(np.column_stack([X[:,0], xintlab]), useranks=True)
        #assert_almost_equal(gs2.groupmeanfilter, stats.rankdata(X[:,0]), 15)

        rankbincount = np.bincount(xranks.astype(int))
        nties = rankbincount[rankbincount > 1]
        ntot = float(len(xranks));
        tiecorrection = 1 - (nties**3 - nties).sum()/(ntot**3 - ntot)
        assert_almost_equal(tiecorrection, stats.tiecorrect(xranks),15)
        print tiecorrection
        print tiecorrect(xranks)

        tot = X.shape[0]
        t=500 #168
        f=(tot*(tot+1.)/12.)-(t/(6.*(tot-1.)))
        f=(tot*(tot+1.)/12.)/stats.tiecorrect(xranks)
        for i,j in zip(v2[diffidx], v1[diffidx]):
            #pdiff = np.abs(mrs[i] - mrs[j])
            pdiff = np.abs(meanranks[i] - meanranks[j])
            se = np.sqrt(f * np.sum(1./xnobs[[i,j]] )) #np.array([8,8]))) #Fixme groupnobs[[i,j]] ))
            print i,j, pdiff, se, pdiff/se, pdiff/se>2.6310

        multicomp = MultiComparison(*X.T)
        multicomp.kruskal()
        gsr = GroupsStats(X, useranks=True)


        for i in range(10):
            x1, x2 = (np.random.randn(30,2) + np.array([0, 0.5])).T
            skw = stats.kruskal(x1, x2)
            mc2=MultiComparison(np.r_[x1, x2], np.r_[np.zeros(len(x1)), np.ones(len(x2))])
            newskw = mc2.kruskal()
            print skw, np.sqrt(skw[0]), skw[1]-newskw, (newskw/skw[1]-1)*100

        tablett, restt, arrtt = multicomp.allpairtest(stats.ttest_ind)
        tablemw, resmw, arrmw = multicomp.allpairtest(stats.mannwhitneyu)
        print tablett
        print tablemw
        tablemwhs, resmw, arrmw = multicomp.allpairtest(stats.mannwhitneyu, method='hs')
        print tablemwhs

    if 'last' in examples:
        xli = (np.random.randn(60,4) + np.array([0, 0, 0.5, 0.5])).T
        #Xrvs = np.array(catstack(xli))
        xrvs, xrvsgr = catstack(xli)
        multicompr = MultiComparison(xrvs, xrvsgr)
        tablett, restt, arrtt = multicompr.allpairtest(stats.ttest_ind)
        print tablett


        xli=[[8,10,9,10,9],[7,8,5,8,5],[4,8,7,5,7]]
        x,l = catstack(xli)
        gs4 = GroupsStats(np.column_stack([x,l]))
        print gs4.groupvarwithin()
