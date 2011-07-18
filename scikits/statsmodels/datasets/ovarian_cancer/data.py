__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

"""Survival data for ovarian cancer"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = ""
SOURCE      = """
This data is taken from Konstantinopoulos PA, Cannistra SA, Fountzilas H,
Culhane A, Pillay K, et al. (2011) [1] and downloaded from [2]

[1] Konstantinopoulos PA, Cannistra SA, Fountzilas H, Culhane A, Pillay K,
    et al. 2011 Integrated Analysis of Multiple Microarray Datasets Identifies
    a Reproducible Survival Predictor in Ovarian Cancer. PLoS ONE 6(3): e18202.
    doi:10.1371/journal.pone.0018202

[2] ncbi.nlm.nih.gov/geo/query/acc.cgi?token=bdgzfwmysouamxe&acc=GSE19161
"""

DESCRSHORT  = """Survival data for ovarian cancer"""

DESCRLONG   = """Survival data for ovarian cancer with
                gene expression covariates, compiled from
                four seperate studies"""

NOTE        = """
Number of Observations: 239
Number of Variables: 660
Variable name definitions:
    time      - survival time (in months)
    censoring - 0 if observation is censored, 1 if observation is
                an event

    The remaining variables are gene expression values identified by
    their names. For details see [1] above.
"""

from numpy import recfromtxt, column_stack, array, genfromtxt, r_
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the survival and gene expression data

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/ovarian_cancer_data.txt', 'rb'),
            dtype=float)
    gene_names = genfromtxt(open(filepath + '/ovarian_gene_names.txt', 'rb'),
            dtype="S11")
    surv_names = array(['time','censoring']).astype("S11")
    names = r_[surv_names,gene_names]
    dataset = Dataset(data=data, names=names)
    return dataset
