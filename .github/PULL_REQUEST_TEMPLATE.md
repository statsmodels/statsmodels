- [ ] closes #xxxx
- [ ] tests added / passed.
- [ ] code/documentation is well formatted.
- [ ] properly formatted commit message. See
      [NumPy's guide](https://docs.scipy.org/doc/numpy-1.15.1/dev/gitwash/development_workflow.html#writing-the-commit-message).

#### AI Disclosure

<!--
REQUIRED. Do not delete or rewrite this section. The "AI Policy Disclosure Check"
CI job fails until exactly one of the two boxes below is ticked ([x]).

This applies to every pull request, including pull requests opened by an AI agent,
opened with `gh pr create --body ...` (which does not use this template), or where
you replace the text above with your own description. In those cases, copy this whole
section, unchanged, into your description.

Keep this in sync with the copy in .github/workflows/ai-policy-check.yml.
-->

Contributions must comply with the statsmodels [AI Policy](https://www.statsmodels.org/devel/dev/ai-policy.html).

Tick **exactly one** of the following. If AI tools were used, replace the `<...>`
placeholders in the second option with the tool name(s) and how the tool's output was
used.

- [ ] No AI tools were used to develop this pull request.
- [ ] AI tools were used.
      Tool(s): `<name(s), e.g. Copilot, Claude, ChatGPT>`.
      Used for: `<e.g. drafting an implementation, writing tests, debugging, editing docstrings>`.
      I have personally read, understood, and can explain every line of this diff, and
      have verified the statistical/numerical correctness of the change.

<details>


**Notes**:

* It is essential that you add a test when making code changes. Tests are not
  needed for doc changes.
* When adding a new function, test values should usually be verified in another package (e.g., R/SAS/Stata).
* When fixing a bug, you must add a test that would produce the bug in main and
  then show that it is fixed with the new code.
* New code additions must be well formatted. Changes should pass ruff. You can
  verify your changes are well formatted by running
  ```
  ruff check . --fix
  ```
  assuming `ruff` is installed. While passing this test is not required, it is good practice and it help
  improve code quality in `statsmodels`.
* Docstring additions must render correctly, including escapes and LaTeX.
* If AI tools were used to help write this PR, see the
  [AI Policy](https://www.statsmodels.org/devel/dev/ai-policy.html) for what disclosure
  and review is expected of you before submitting.

</details>
