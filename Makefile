python := /bin/env python

build: FORCE
	$(python) setup.py build

install: FORCE
	$(python) setup.py install

test: test-install FORCE
	PYTHONPATH=.scratchlib && $(python) .scratchlib/neuroimaging/tests/test_all.py

test-install: FORCE
	$(python) setup.py install --install-lib=.scratchlib

clean: FORCE
	./clean

FORCE:
