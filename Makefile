python := /bin/env python
#version := $(PYTHONPATH=./lib && scripts/nipy-doc --version)
version := 0.1
package_name := python-nipy
rpm_name := $(package_name)-$(version)
checkinstall := /usr/sbin/checkinstall
docs_dir := $(shell pwd)/doc

build: FORCE
	$(python) setup.py build

install: FORCE
	$(python) setup.py install

test: FORCE
	./test

test-install: FORCE
	$(python) setup.py install --install-lib=test-install

clean: docs-clean FORCE
	./clean

FORCE:

include $(docs_dir)/Makefile
