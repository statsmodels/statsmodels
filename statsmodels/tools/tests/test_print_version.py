from contextlib import redirect_stdout
import io

from statsmodels.tools.print_version import _show_versions_only, show_versions


def test_show_versions_show_dirs_false_does_not_print_full_listing():
    # GH: show_versions(show_dirs=False) called _show_versions_only() but
    # had no return, so it fell through and printed the full listing
    # (including install directories) anyway.
    buf_only = io.StringIO()
    with redirect_stdout(buf_only):
        _show_versions_only()

    buf_show = io.StringIO()
    with redirect_stdout(buf_show):
        show_versions(show_dirs=False)

    assert buf_show.getvalue() == buf_only.getvalue()
