#collect some imports of verified (at least one example) functions
from statsmodels.sandbox.stats.multicomp import \
             multipletests, fdrcorrection0, fdrcorrection_twostage, tukeyhsd
#==============================================
#
# Part 1: Multiple Tests and P-Value Correction
#
#==============================================


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
