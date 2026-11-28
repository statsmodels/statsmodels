#!/usr/bin/env python
"""
Collect the release-note fragments into the release notes.

Each pull request adds its release-note bullets as a fragment file in
``docs/upcoming_changes`` instead of editing the release notes, so that pull
requests do not conflict there. See ``docs/upcoming_changes/README.rst``.

Usage::

    python tools/releasing/collect_release_notes.py --check
    python tools/releasing/collect_release_notes.py docs/source/release/version0.16.0.rst
    python tools/releasing/collect_release_notes.py --remove docs/source/release/version0.16.0.rst

``--check`` validates the fragments. Given a release notes file, the bullets
of each fragment are appended to the end of the list of their section, in
the order of the pull request numbers. ``--remove`` also deletes the
collected fragments.
"""

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
FRAGMENTS = ROOT / "docs" / "upcoming_changes"

# Fragment key -> (top-level heading, subsection heading or None)
SECTIONS = {
    "deprecation": ("Breaking Changes and Deprecations", "Deprecations"),
    "behavior": ("Breaking Changes and Deprecations", "Behavior changes"),
    "feature": ("New Features and Enhancements", "New functions and methods"),
    "performance": ("New Features and Enhancements", "Performance"),
    "regression": ("Bug Fixes", "Regression and linear models"),
    "glm": ("Bug Fixes", "Generalized linear models and discrete models"),
    "mixed": ("Bug Fixes", "Mixed models and imputation"),
    "tsa": ("Bug Fixes", "Time series analysis"),
    "stats": ("Bug Fixes", "Statistics and tests"),
    "treatment": ("Bug Fixes", "Treatment effects"),
    "robust": ("Bug Fixes", "Robust and multivariate statistics"),
    "nonparametric": ("Bug Fixes", "Nonparametric"),
    "duration": ("Bug Fixes", "Duration models"),
    "other": ("Bug Fixes", "Other"),
    "documentation": ("Documentation", None),
    "validation-confidence": (
        "Stricter Input Validation",
        "Confidence level and significance",
    ),
    "validation-counts": (
        "Stricter Input Validation",
        "Counts, sample sizes, weights and probabilities",
    ),
    "validation-intervals": (
        "Stricter Input Validation",
        "Intervals, tuning parameters and shapes",
    ),
    "validation-tsa": ("Stricter Input Validation", "Time series"),
}

NAME = re.compile(r"^(?P<pr>\d+)\.(?P<key>[a-z-]+)\.rst$")
PR_ROLE = re.compile(r":pr:`(\d+)`")


def fragments():
    """The fragment files as (pr, key, path), sorted by pull request."""
    out = []
    for path in sorted(FRAGMENTS.glob("*.rst")):
        if path.name == "README.rst":
            continue
        match = NAME.match(path.name)
        key = match.group("key") if match else None
        pr = int(match.group("pr")) if match else -1
        out.append((pr, key, path))
    return sorted(out, key=lambda item: (item[0], item[1] or "", item[2].name))


def check(items):
    """Problems with the fragments, as a list of messages."""
    problems = []
    for pr, key, path in items:
        name = path.name
        if pr < 0 or key not in SECTIONS:
            problems.append(
                f"{name}: the name must be <pr number>.<key>.rst with a key "
                f"in {sorted(SECTIONS)}"
            )
            continue
        raw = path.read_bytes()
        if b"\r" in raw:
            problems.append(f"{name}: use LF line endings")
        text = raw.decode("utf-8")
        lines = text.rstrip("\n").split("\n")
        if not lines or not lines[0].startswith("- "):
            problems.append(f"{name}: the first line must start a bullet ('- ')")
        for i, line in enumerate(lines, 1):
            if line != line.rstrip():
                problems.append(f"{name}:{i}: trailing whitespace")
            if line and not line.startswith(("- ", "  ")):
                problems.append(
                    f"{name}:{i}: continuation lines are indented by 2 spaces"
                )
        for bullet in _bullets(lines):
            prs = PR_ROLE.findall(" ".join(bullet))
            if str(pr) not in prs:
                problems.append(
                    f"{name}: every bullet needs the role :pr:`{pr}`"
                )
                break
    return problems


def _bullets(lines):
    bullet = []
    for line in lines:
        if line.startswith("- ") and bullet:
            yield bullet
            bullet = []
        if line:
            bullet.append(line)
    if bullet:
        yield bullet


def _headings(lines):
    """(index of title line, title, underline character) for each heading."""
    out = []
    for i in range(len(lines) - 1):
        title, under = lines[i], lines[i + 1]
        if (
            title.strip()
            and under
            and len(set(under)) == 1
            and under[0] in "=-~^"
            and len(under) >= len(title.strip())
        ):
            out.append((i, title.strip(), under[0]))
    return out


def _section_end(lines, heads, top, sub):
    """Index after the last non-blank line of a section's body."""
    top_idx = [n for n, (_, t, c) in enumerate(heads) if t == top and c == "="]
    if not top_idx:
        raise KeyError(f"heading {top!r} not found")
    start = top_idx[0]
    if sub is not None:
        for n in range(start + 1, len(heads)):
            if heads[n][2] == "=":
                break
            if heads[n][1] == sub:
                start = n
                break
        else:
            raise KeyError(f"heading {sub!r} under {top!r} not found")
    # the body runs from below the underline to the next heading of any level
    body_start = heads[start][0] + 2
    end = heads[start + 1][0] if start + 1 < len(heads) else len(lines)
    last = body_start
    for i in range(body_start, end):
        if lines[i].strip():
            last = i + 1
    return last


def collect(notes_path, items):
    """Insert the fragments' bullets into the release notes file."""
    notes = Path(notes_path)
    lines = notes.read_bytes().decode("utf-8").split("\n")
    by_key = {}
    for _, key, path in items:
        text = path.read_bytes().decode("utf-8").rstrip("\n")
        by_key.setdefault(key, []).extend(text.split("\n"))
    # insert from the bottom of the file up, so earlier indices stay valid
    targets = []
    for key, new in by_key.items():
        top, sub = SECTIONS[key]
        heads = _headings(lines)
        targets.append((_section_end(lines, heads, top, sub), key, new))
    for at, _, new in sorted(targets, key=lambda t: t[0], reverse=True):
        lines[at:at] = new
    notes.write_bytes("\n".join(lines).encode("utf-8"))
    return {key: len(new) for _, key, new in targets}


def render_pending(items):
    """
    reST for the fragments that are not yet in the release notes.

    The documentation build includes this at the end of the release notes of
    the release in progress, so that the development documentation shows
    every change. Once the fragments are collected, it is only a comment.
    """
    lines = [
        ".. Generated from docs/upcoming_changes by docs/source/conf.py.",
        "",
    ]
    by_key = {}
    for _, key, path in items:
        if key in SECTIONS:
            text = path.read_bytes().decode("utf-8").rstrip("\n")
            by_key.setdefault(key, []).extend(text.split("\n"))
    if not by_key:
        return "\n".join(lines) + "\n"
    title = "Changes not yet collected into the notes above"
    lines += [
        title,
        "=" * len(title),
        "",
        (
            "These changes were merged after the notes above were last "
            "updated. They are moved into the sections above when the release "
            "is made."
        ),
        "",
    ]
    for key, (top, sub) in SECTIONS.items():
        if key not in by_key:
            continue
        head = f"{sub} ({top})" if sub else top
        lines += [head, "-" * len(head), "", *by_key[key], ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("notes", nargs="?", help="release notes file to update")
    parser.add_argument("--check", action="store_true", help="only validate")
    parser.add_argument(
        "--remove", action="store_true", help="delete the collected fragments"
    )
    args = parser.parse_args(argv)
    items = fragments()
    problems = check(items)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    if args.check or args.notes is None:
        print(f"{len(items)} fragment(s) OK")
        return 0
    counts = collect(args.notes, items)
    for key, n in sorted(counts.items()):
        print(f"{key}: {n} line(s) added")
    if args.remove:
        for _, _, path in items:
            path.unlink()
        print(f"removed {len(items)} fragment(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
