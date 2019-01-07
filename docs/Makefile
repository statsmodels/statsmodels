# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = statsmodels
SOURCEDIR     = source
BUILDDIR      = build

PAPER         =
TOOLSPATH	  = ../tools/
DATASETBUILD  = dataset_rst.py
EXAMPLEBUILD  = examples_rst.py
NOTEBOOKBUILD = nbgenerate.py
FOLDTOC  = fold_toc.py

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS)


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)

.PHONY: help Makefile

cleanall: 
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)
	-rm source/examples/generated/*
	-rm -rf source/examples/notebooks/generated/*
	-rm -rf ../tools/hash_dict.pickle
	-rm -rf source/datasets/generated/*

notebooks:
	@echo "Generating notebooks from examples/notebooks folder"
	$(TOOLSPATH)$(NOTEBOOKBUILD) --execute=True --allow_errors=True

html:
    # make directories for images
	@echo "Make static directory for images"
	mkdir -p $(BUILDDIR)/html/_static
	# generate the examples rst files
	@echo "Generating reST from examples folder"
	#$(TOOLSPATH)$(EXAMPLEBUILD)
	@echo "Generating datasets from installed statsmodels.datasets"
	$(TOOLSPATH)$(DATASETBUILD)
	@echo "Generating notebooks from examples/notebooks folder"
	$(TOOLSPATH)$(NOTEBOOKBUILD) --parallel --report-errors --skip-existing
	@echo "Running sphinx-build"
	@echo @$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)
	@echo "Copying rendered example notebooks"
	mkdir -p $(BUILDDIR)/html/examples/notebooks/generated
	cp source/examples/notebooks/generated/*html $(BUILDDIR)/html/examples/notebooks/generated
	@echo $(TOOLSPATH)$(FOLDTOC) $(BUILDDIR)/html/index.html
	$(TOOLSPATH)$(FOLDTOC) $(BUILDDIR)/html/index.html
	@echo $(TOOLSPATH)$(FOLDTOC) $(BUILDDIR)/html/dev/index.html ../_static
	$(TOOLSPATH)$(FOLDTOC) $(BUILDDIR)/html/dev/index.html ../_static

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)
