# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

PAPER         =
TOOLSPATH	  = ../tools/
DATASETBUILD  = dataset_rst.py
NOTEBOOKBUILD = nbgenerate.py

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html:
    # make directories for images
	@echo "Make static directory for images"
	mkdir -p $(BUILDDIR)/html/_static
	# generate the examples rst files
	@echo "Generating datasets from installed statsmodels.datasets"
	$(TOOLSPATH)$(DATASETBUILD)
	@echo "Executing notebooks from examples/notebooks folder"
	mkdir -p $(BUILDDIR)/source/examples/notebooks/generated
	# Black list notebooks from doc build here
	$(TOOLSPATH)$(NOTEBOOKBUILD) --parallel --report-errors --skip-existing --execute-only --execution-blacklist statespace_custom_models
	@echo "Running sphinx-build"
	@echo @$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(O)
