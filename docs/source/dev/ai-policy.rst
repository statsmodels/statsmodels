.. _ai-policy-label:

AI Policy
=========

“AI” herein refers to generative AI tools such as large language models
(LLMs) and AI coding assistants (e.g. Copilot, ChatGPT, Claude) that can
generate, edit, or review software code, tests, or documentation, or
generate human-like text and communication. This policy applies to all
contributions to statsmodels, including issues, pull requests, code
review comments, and mailing list / discussion posts.

statsmodels welcomes the responsible use of AI tools to help
contributors write patches. AI can lower the barrier to contributing and
speed up routine work, but it does not change any of statsmodels’
existing standards for correctness, style, testing, or review. The use
of AI is a matter for disclosure and quality, not a shortcut around
them.

Responsibility
--------------

You are responsible for any contribution you submit, regardless of
whether it was written by hand or generated, in whole or in part, with
AI. AI is a tool to help you produce a patch more quickly; it is not a
substitute for your own understanding of the problem, and it does not
relieve you of responsibility for the correctness of the fix.

Before opening a pull request you must:

- Personally read, understand, and be able to explain every line of the
  diff, including any surrounding code it touches.
- Verify that the statistics, econometrics, or numerical methods
  involved are correct — AI tools frequently produce code that runs
  without error but that implements the wrong formula, uses an
  inappropriate estimator, mishandles missing data or edge cases, or
  silently changes model semantics. Passing tests is necessary but not
  sufficient; you are expected to reason about whether the result is
  statistically correct, ideally cross-checked against a reference
  implementation (e.g. R, SAS, Stata, or the relevant paper) as
  described in the main `contributing
  guide <https://www.statsmodels.org/stable/dev/index.html>`__.
- Confirm the change is consistent with statsmodels’ existing API
  conventions and design.

It is not acceptable to submit a patch that you cannot understand and
explain yourself. PRs where the author cannot answer reviewers’
questions about their own submitted code will be closed.

Disclosure
----------

You must disclose whether AI was used to assist in developing a pull
request. If so, state in the PR description which tool(s) were used
(e.g. “Copilot”, “Claude”, “ChatGPT”) and roughly how they were used
(e.g. drafting an implementation, writing tests, debugging, or editing
docstrings). Pull requests that do not include this disclosure will be
rejected.

Disclosure is about transparency, not permission-seeking: using AI is
fine, but reviewers need to know so they can calibrate their review
accordingly.

Code Quality
------------

Contributions are held to the same bar regardless of how they were
produced. We will reject pull requests that we deem to be “AI slop” —
code that is fully or largely AI-generated and does not meet
statsmodels’ standards, or that was submitted without the review
described above. Do not waste maintainers’ time with shallow,
unreviewed, or exploratory AI output.

Concretely, every pull request must:

- **Lint cleanly.** All Python code must pass both ``ruff`` and
  ``flake8`` with the project’s configuration, with no new warnings or
  errors. You can check your diff locally, for example:

  .. code:: bash

     ruff check .
     git diff upstream/main -u -- "*.py" | flake8 --diff --isolated

  CI will fail on lint errors, and PRs with unresolved lint failures will not be reviewed.

- **Follow the project’s coding standards**, including docstring
  conventions (numpydoc style) and existing patterns in the surrounding
  module. Docstring changes must render correctly, including math and
  LaTeX.

Testing
-------

Every code change must be fully tested, and it is the submitter’s — not
the reviewer’s or maintainer’s — responsibility to ensure the fix is
correct.

- All new code and bug fixes require tests. Tests are not required only
  for documentation-only changes.
- Changes must be covered by **full test coverage**: every new branch
  and line introduced or modified by the PR should be exercised by the
  test suite. Use ``pytest --cov`` to confirm coverage locally before
  opening the PR.
- When fixing a bug, include a regression test that fails against
  ``main`` and passes with your change.
- When adding a new statistical function, method, or model, its results
  should be validated against an independent, trusted source (e.g. R,
  SAS, Stata, or a published reference), and the test should encode that
  comparison rather than simply asserting that the code runs. Where external
  validation has been used to produce reference values, include a comment
  in the test explaining the source of those values. Ideally the R/Stata/SAS
  code used to generate the reference values should be included in the test
  file as a comment.
- If you used AI to help write tests, review them with particular care:
  a common failure mode is tests that merely restate the implementation
  (and so would pass even if the implementation is wrong) rather than
  testing against an independent expectation.

A PR that passes CI but was not meaningfully tested by a human against a
trusted reference does not meet this bar, even if AI wrote the tests.

Copyright
---------

All code in statsmodels is released under the `modified BSD (3-clause)
license <https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt>`__.
Contributors license their code under the same terms when it is merged
into statsmodels. You must own the copyright of any code you submit, or
include the BSD-3-clause-compatible license(s) that apply to it in the
patch.

Code generated by AI tools may reproduce material from their training
data in ways that infringe copyright, and it is the submitter’s
responsibility to ensure this is not the case. We reserve the right to
reject any pull request, AI-assisted or not, where copyright is in
question.

Communication
-------------

When interacting with other contributors — in issues, pull requests,
code review, or on the mailing list — do not use AI to write on your
behalf, except for translation or grammar/spelling correction if you are
not a native English speaker. Please don’t paste raw AI output into
issue or PR descriptions or review comments: it makes it harder for
maintainers to assess the substance of what you’re saying, and it isn’t
a substitute for engaging with reviewer feedback yourself. We want to
interact with the human behind the contribution, not a chatbot speaking
for them.

AI Agents and Automated Contributions
-------------------------------------

The use of a fully autonomous AI agent that writes code and opens a pull
request without a human reviewing it first is not permitted. A human
must review any AI-assisted output and take responsibility for it as
described above before it is submitted.

Please refrain from submitting issues or pull requests generated by
fully-automated tools with no meaningful human review. Maintainers
reserve the right, at their sole discretion, to close such submissions
without review and to block the account responsible for them.

Acknowledgements
----------------

This policy draws on the published AI policies of
`NumPy <https://numpy.org/devdocs/dev/ai_policy.html>`__,
`pandas <https://pandas.pydata.org/docs/dev/development/contributing.html#automated-contributions-policy>`__,
`scikit-learn <https://scikit-learn.org/stable/developers/contributing.html#automated-contributions-policy>`__,
and SciPy. We thank those projects and communities for their work in
this area.