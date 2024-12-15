:orphan:

==============
Release 0.13.2
==============

Release summary
===============

statsmodels is using github to store the updated documentation. Two version are available:

- `Stable <https://www.statsmodels.org/>`_, the latest release
- `Development <https://www.statsmodels.org/devel/>`_, the latest build of the main branch

**Warning**

API stability is not guaranteed for new features, although even in
this case changes will be made in a backwards compatible way if
possible. The stability of a new feature depends on how much time it
was already in statsmodels main and how much usage it has already
seen.  If there are specific known problems or limitations, then they
are mentioned in the docstrings.

Stats
-----
**Issues Closed**: 61

**Pull Requests Merged**: 35


What's new - an overview
========================
This a bug fix and deprecation only release.

Submodules
----------

``backport maintenance/0.13.x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- [maintenance/0.13.x] Merge pull request #7991 from ChadFulton/ss-exp-smth-seasonals  (:pr:`8062`)



``maintenance``
~~~~~~~~~~~~~~~
- [maintenance/0.13.x] Merge pull request #7989 from bashtage/try-oldest-supported-numpy  (:pr:`8054`)
- [maintenance/0.13.x] Merge pull request #7906 from bashtage/reverse-seasonal  (:pr:`8055`)
- [maintenance/0.13.x] Merge pull request #7939 from bashtage/test-pandas-compat  (:pr:`8058`)
- [maintenance/0.13.x] Merge pull request #8000 from bashtage/unsigned-int-comparrison  (:pr:`8064`)
- [maintenance/0.13.x] Merge pull request #8003 from pkaf/ets-loglike-doc  (:pr:`8065`)
- [maintenance/0.13.x] Merge pull request #8007 from rambam613/patch-1  (:pr:`8066`)
- [maintenance/0.13.x] Merge pull request #8015 from ChadFulton/ss-docs  (:pr:`8068`)
- [maintenance/0.13.x] Merge pull request #8023 from MichaelChirico/patch-1  (:pr:`8069`)
- [maintenance/0.13.x] Merge pull request #8026 from wirkuttis/bugfix_statstools  (:pr:`8070`)
- [maintenance/0.13.x] Merge pull request #8047 from bashtage/fix-lowess-8046  (:pr:`8073`)
- Correct upstream target  (:pr:`8074`)
- [maintenance/0.13.x] Merge pull request #7916 from zprobs/main  (:pr:`8075`)
- [maintenance/0.13.x] Merge pull request #8037 from bashtage/future-pandas  (:pr:`8077`)
- [maintenance/0.13.x] Merge pull request #8004 from bashtage/doc-slim  (:pr:`8079`)
- [maintenance/0.13.x] Merge pull request #7946 from bashtage/remove-looseversion  (:pr:`8082`)
- Cleanup CI  (:pr:`8083`)
- [maintenance/0.13.x] Merge pull request #7950 from bashtage/cond-number  (:pr:`8084`)
- Correct backport errors  (:pr:`8085`)
- Correct small future issues  (:pr:`8089`)
- Correct setup for oldest supported  (:pr:`8092`)


Development summary and credits
===============================

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance for this release came from

- Chad Fulton
- Brock Mendel
- Peter Quackenbush
- Kerby Shedden
- Kevin Sheppard

and the general maintainer and code reviewer

- Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.13.2 release (based on git log):

- Chad Fulton
- Josef Perktold
- Kevin Sheppard


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`8053`: [maintenance/0.13.x] Merge pull request #8035 from swallan/scipy-studentized-range-qcrit-pvalue
- :pr:`8054`: [maintenance/0.13.x] Merge pull request #7989 from bashtage/try-oldest-supported-numpy
- :pr:`8055`: [maintenance/0.13.x] Merge pull request #7906 from bashtage/reverse-seasonal
- :pr:`8056`: [maintenance/0.13.x] Merge pull request #7921 from bashtage/mean-diff-plot
- :pr:`8057`: [maintenance/0.13.x] Merge pull request #7927 from bashtage/enricovara-patch-1
- :pr:`8058`: [maintenance/0.13.x] Merge pull request #7939 from bashtage/test-pandas-compat
- :pr:`8059`: [maintenance/0.13.x] Merge pull request #7954 from bashtage/recursive-ls-heading
- :pr:`8060`: [maintenance/0.13.x] Merge pull request #7969 from bashtage/hw-wrong-param
- :pr:`8061`: [maintenance/0.13.x] Merge pull request #7988 from bashtage/relax-tol-var-test
- :pr:`8062`: [maintenance/0.13.x] Merge pull request #7991 from ChadFulton/ss-exp-smth-seasonals
- :pr:`8063`: [maintenance/0.13.x] Merge pull request #7995 from bashtage/remove-aliasing
- :pr:`8064`: [maintenance/0.13.x] Merge pull request #8000 from bashtage/unsigned-int-comparrison
- :pr:`8065`: [maintenance/0.13.x] Merge pull request #8003 from pkaf/ets-loglike-doc
- :pr:`8066`: [maintenance/0.13.x] Merge pull request #8007 from rambam613/patch-1
- :pr:`8068`: [maintenance/0.13.x] Merge pull request #8015 from ChadFulton/ss-docs
- :pr:`8069`: [maintenance/0.13.x] Merge pull request #8023 from MichaelChirico/patch-1
- :pr:`8070`: [maintenance/0.13.x] Merge pull request #8026 from wirkuttis/bugfix_statstools
- :pr:`8072`: [maintenance/0.13.x] Merge pull request #8042 from bashtage/pin-numpydoc
- :pr:`8073`: [maintenance/0.13.x] Merge pull request #8047 from bashtage/fix-lowess-8046
- :pr:`8074`: MAINT: Correct upstream target
- :pr:`8075`: [maintenance/0.13.x] Merge pull request #7916 from zprobs/main
- :pr:`8077`: [maintenance/0.13.x] Merge pull request #8037 from bashtage/future-pandas
- :pr:`8078`: [maintenance/0.13.x] Merge pull request #8005 from bashtage/mle-results-doc
- :pr:`8079`: [maintenance/0.13.x] Merge pull request #8004 from bashtage/doc-slim
- :pr:`8080`: [maintenance/0.13.x] Merge pull request #7875 from ZachariahPang/Fix-wrong-order-datapoints
- :pr:`8081`: [maintenance/0.13.x] Merge pull request #7940 from bashtage/future-co…
- :pr:`8082`: [maintenance/0.13.x] Merge pull request #7946 from bashtage/remove-looseversion
- :pr:`8083`: MAINT: Cleanup CI
- :pr:`8084`: [maintenance/0.13.x] Merge pull request #7950 from bashtage/cond-number
- :pr:`8085`: MAINT: Correct backport errors
- :pr:`8088`: MAINT: Stop using conda temporarily
- :pr:`8089`: MAINT: Correct small future issues
- :pr:`8092`: MAINT: Correct setup for oldest supported
- :pr:`8096`: [maintenance/0.13.x] Merge pull request #8093 from josef-pkt/bug_proportion_pwer_2indep
- :pr:`8097`: [maintenance/0.13.x] Merge pull request #8086 from xjcl/patch-1
