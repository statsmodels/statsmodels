Development Workflow for Maintainers
------------------------------------

Git Workflow
~~~~~~~~~~~~~

Releasing
~~~~~~~~~

#. Fix the version number.
#. Tag the release. For example::

    git tag -a v0.3.0rc1 -m "Version 0.3.0 Release Candidate 1" 7b2fb295a421b83a90b04180c8a1678cf9a6ed0d


Commit Comments
~~~~~~~~~~~~~~~
Prefix commit messages in the master branch of the shared repository with the following::

    ENH: Feature implementation
    BUG: Bug fix
    STY: Coding style changes (indenting, braces, code cleanup)
    DOC: Sphinx documentation, docstring, or comment changes
    CMP: Fixing compiled code issues, regenerating C code with Cython, etc.
