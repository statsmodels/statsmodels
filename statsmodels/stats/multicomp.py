#collect some imports of verified (at least one example) functions
from statsmodels.sandbox.stats.multicomp import \
             multipletests, fdrcorrection0, fdrcorrection_twostage, tukeyhsd
#==============================================
#
# Part 1: Multiple Tests and P-Value Correction
#
#==============================================

def ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

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
    new procedure 'fdr_gbs' not verified yet, p-values derived from scratch not
    reference

    All procedures that are included, control FWER or FDR in the independent
    case, and most are robust in the positively correlated case.

    fdr_gbs: high power, fdr control for independent case and only small
    violation in positively correlated case


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

    elif method.lower() in ['fdr_gbs']:
        #adaptive stepdown in Favrilov, Benjamini, Sarkar, Annals of Statistics 2009
##        notreject = pvals > alphaf / np.arange(ntests, 0, -1) #alphacSidak
##        notrejectmin = np.min(np.nonzero(notreject))
##        notreject[notrejectmin:] = True
##        reject = ~notreject

        ii = np.arange(1, ntests + 1)
        q = (ntests + 1. - ii)/ii * pvals / (1. - pvals)
        pvals_corrected_raw = np.maximum.accumulate(q) #up requirementd

        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        reject = pvals_corrected < alpha

    else:
        raise ValueError('method not recognized')


    if not pvals_corrected is None: #not necessary anymore
        pvals_corrected[pvals_corrected>1] = 1
    if returnsorted:
        return reject, pvals_corrected, alphacSidak, alphacBonf
    else:
        if pvals_corrected is None:
            return reject[sortrevind], pvals_corrected, alphacSidak, alphacBonf
        else:
            return reject[sortrevind], pvals_corrected[sortrevind], alphacSidak, alphacBonf

#TODO: rename drop 0 at end
def fdrcorrection0(pvals, alpha=0.05, method='indep'):
    '''pvalue correction for false discovery rate

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests. Both are
    available in the function multipletests, as method=`fdr_bh`, resp. `fdr_by`.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : {'indep', 'negcorr')

    Returns
    -------
    rejected : array, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----

    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to alpha * m/m_0 where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Method names can be abbreviated to first letter, 'i' or 'p' for fdr_bh and 'n' for
    fdr_by.



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

def fdrcorrection_twostage(pvals, alpha=0.05, iter=False):
    '''(iterated) two stage linear step-up procedure with estimation of number of true
    hypotheses

    Benjamini, Krieger and Yekuteli, procedure in Definition 6

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : {'indep', 'negcorr')

    Returns
    -------
    rejected : array, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : array
        pvalues adjusted for multiple hypotheses testing to limit FDR
    m0 : int
        ntest - rej, estimated number of true hypotheses
    alpha_stages : list of floats
        A list of alphas that have been used at each stage

    Notes
    -----
    The returned corrected p-values, are from the last stage of the fdr_bh linear step-up
    procedure (fdrcorrection0 with method='indep')

    BKY described several other multi-stage methods, which would be easy to implement.
    However, in their simulation the simple two-stage method (with iter=False) was the
    most robust to the presence of positive correlation

    TODO: What should be returned?

    '''
    ntests = len(pvals)
    alpha_prime = alpha/(1+alpha)
    rej, pvalscorr = fdrcorrection0(pvals, alpha=alpha_prime, method='indep')
    r1 = rej.sum()
    if (r1 == 0) or (r1 == ntests):
        return rej, pvalscorr, ntests - r1
    ri_old = r1
    alpha_stages = [alpha_prime]
    while 1:
        ntests0 = ntests - ri_old
        alpha_star = alpha_prime * ntests / ntests0
        alpha_stages.append(alpha_star)
        #print ntests0, alpha_star
        rej, pvalscorr = fdrcorrection0(pvals, alpha=alpha_star, method='indep')
        ri = rej.sum()
        if (not iter) or ri == ri_old:
            break
        elif ri < ri_old:
            raise RuntimeError(" oops - shouldn't be here")
        ri_old = ri

    return rej, pvalscorr, ntests - ri, alpha_stages
