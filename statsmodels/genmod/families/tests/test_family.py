"""
Test functions for genmod.families.family
"""
import pytest

import statsmodels.genmod.families as F
import statsmodels.genmod.families.links as L

all_links = {
    L.Logit, L.logit, L.Power, L.inverse_power, L.sqrt, L.inverse_squared,
    L.identity, L.Log, L.log, L.CDFLink, L.probit, L.cauchy, L.LogLog,
    L.loglog, L.CLogLog, L.cloglog, L.NegativeBinomial, L.nbinom
}
poisson_links = {L.Log, L.log, L.identity, L.sqrt}
gaussian_links = {L.Log, L.log, L.identity, L.inverse_power}
gamma_links = {L.Log, L.log, L.identity, L.inverse_power}
binomial_links = {
    L.Logit, L.logit, L.probit, L.cauchy, L.Log, L.log, L.CLogLog,
    L.cloglog, L.LogLog, L.loglog, L.identity
}
inverse_gaussian_links = {
    L.inverse_squared, L.inverse_power, L.identity, L.Log, L.log
}
negative_bionomial_links = {
    L.Log, L.log, L.CLogLog, L.cloglog, L.identity, L.NegativeBinomial,
    L.nbinom, L.Power
}
tweedie_links = {L.Log, L.log, L.Power}

link_cases = [
    (F.Poisson, poisson_links),
    (F.Gaussian, gaussian_links),
    (F.Gamma, gamma_links),
    (F.Binomial, binomial_links),
    (F.InverseGaussian, inverse_gaussian_links),
    (F.NegativeBinomial, negative_bionomial_links),
    (F.Tweedie, tweedie_links)
]


@pytest.mark.parametrize("family, links", link_cases)
def test_invalid_family_link(family, links):
    invalid_links = all_links - links
    with pytest.raises(ValueError):
        for link in invalid_links:
            family(link())


@pytest.mark.parametrize("family, links", link_cases)
def test_family_link(family, links):
    for link in links:
        assert family(link())
