- [ ] closes #xxxx
- [ ] tests added / passed. 
- [ ] code/documentation is well formatted.  
- [ ] properly formatted commit message. See 
      [NumPy's guide](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html#writing-the-commit-message). 

**Notes**:

* It is essential that you add a test when making code changes. Tests are not 
  needed for doc changes.
* When adding a new function, test values should usually be verified in another package (e.g., R/SAS/Stata).
* When fixing a bug, you must add a test that would produce the bug in master and then show that it is fixed with the new code.
* New code additions must be well formatted. Changes should pass flake8.
* Docstring additions must render correctly, including escapes and LaTeX.
