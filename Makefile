python := /bin/env python

build: FORCE
	$(python) setup.py build

install: FORCE
	$(python) setup.py install

test: FORCE
	$(python) ./test

test-install: FORCE
	$(python) setup.py install --install-lib=test-install

clean: FORCE
	./clean

FORCE:
