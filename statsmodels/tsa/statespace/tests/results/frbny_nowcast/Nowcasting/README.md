The code in this folder is based on the Federal Reserve Bank of New York code
found at https://github.com/FRBNY-TimeSeriesAnalysis/Nowcasting, which was
downloaded as of commit 19f365cab8269e3aac3faa11ad091d6e913c5c43. Only the
files from that repository which were required for generating the test results
are included here.

In additionm the following files from the original package have been modified
(use git diff against the above repository to see the changes)

- functions/dfm.m
- functions/update_nowcast.m

The following files are not a part of the original package:

- test_DFM_blocks.m
- test_DFM.m
- test_news_blocks.m
- test_news.m
- test_spec_blocks.xls
- test_spec.xls
