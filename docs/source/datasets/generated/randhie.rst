RAND Health Insurance Experiment Data
=====================================

Description
-----------



Notes
-----

Number of observations - 20,190
Number of variables - 10
Variable name definitions::

    mdvis   - Number of outpatient visits to an MD
    lncoins - ln(coinsurance + 1), 0 <= coninsurance <= 100
    idp     - 1 if individual deductible plan, 0 otherwise
    lpi     - ln(max(1, annual participation incentive payment))
    fmde    - 0 if idp = 1; ln(max(1, MDE/(0.01 coinsurance))) otherwise
    physlm  - 1 if the person has a physical limitation
    disea   - number of chronic diseases
    hlthg   - 1 if self-rated health is good
    hlthf   - 1 if self-rated health is fair
    hlthp   - 1 if self-rated health is poor
    (Omitted category is excellent self-rated health)


Source
------

The data was collected by the RAND corporation as part of the Health
Insurance Experiment (HIE).

http://www.rand.org/health/projects/hie/

This data was used in::

    Cameron, A.C. amd Trivedi, P.K. 2005.  `Microeconometrics: Methods
        and Applications,` Cambridge: New York.

And was obtained from: <http://cameron.econ.ucdavis.edu/mmabook/mmadata.html>

See randhie/src for the original data and description.  The data included
here contains only a subset of the original data.  The data varies slightly
compared to that reported in Cameron and Trivedi.


Copyright
---------

This is in the public domain.
