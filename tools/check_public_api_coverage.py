"""
Check that the public API surface (statsmodels.*.api exports, plus every
name listed in a docs/source/*.rst autosummary block, expanded to public
methods and properties on any class) is exercised by the test suite.

This measures each object's *body* coverage, not its `def` line -- the def
line always executes at import time and would make every function look
partly covered regardless of whether it is ever called.

Usage
-----
1. Produce a coverage.json file from a full test run, e.g.::

       pytest statsmodels -n auto -m "not example" \\
           --cov=statsmodels --cov-config=tools/coverage_public_api.cfg \\
           --cov-report=json:coverage.json

   ``tools/coverage_public_api.cfg`` should set ``branch = False`` and omit
   ``tests/``, ``sandbox/`` and ``examples/`` -- see that file for the
   config used to produce the baseline.

2. Run this script, pointing at the coverage.json file::

       python tools/check_public_api_coverage.py coverage.json

   With no baseline file, it prints every zero-coverage public object.
   With ``--baseline tools/public_api_coverage_baseline.json``, it instead
   fails (exit code 1) only on public objects that are zero-coverage now
   but were NOT in the baseline -- i.e. new gaps -- so this can be wired
   into CI as a check that stops the untested surface from growing without
   requiring the pre-existing gaps to be fixed first.

   Pass ``--write-baseline PATH`` to (re)generate a baseline file from the
   current state, e.g. after deliberately closing some gaps.

Background
----------
This script exists because a full audit of statsmodels 0.15.0.dev found
that several public, documented functions had been silently broken for
years (raising AttributeError on any call) purely because nothing in the
test suite ever executed them -- see e.g. the fixes to
statsmodels/tsa/filters/filtertools.py and
statsmodels/stats/multivariate_tools.py. Zero test coverage on a public
object is a leading indicator of exactly this kind of rot.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import pathlib
import re
import sys
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG_ROOT = ROOT / "statsmodels"
DOCS_ROOT = ROOT / "docs" / "source"


# --------------------------------------------------------------------------
# 1. Resolve the public API surface: api.py exports + autosummary entries.
# --------------------------------------------------------------------------

def _find_api_modules() -> list[str]:
    mods = []
    for p in sorted(PKG_ROOT.rglob("api.py")):
        rel = p.relative_to(PKG_ROOT.parent)
        mod = ".".join(rel.with_suffix("").parts)
        if "sandbox" in mod:
            continue
        mods.append(mod)
    mods.append("statsmodels.formula.api")
    return sorted(set(mods))


_CUR_MOD_RE = re.compile(r"^\.\.\s+(?:currentmodule|module)::\s+(\S+)")
_AUTOSUM_RE = re.compile(r"^(\s*)\.\.\s+autosummary::")
_OPTION_RE = re.compile(r"^\s*:\w[\w-]*:")


def _parse_documented_names() -> list[tuple[str | None, str]]:
    """Return (currentmodule, name) pairs from every autosummary block."""
    entries = []
    for rst in sorted(DOCS_ROOT.rglob("*.rst")):
        cur = None
        lines = rst.read_text(encoding="utf8", errors="replace").split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _CUR_MOD_RE.match(line)
            if m:
                cur = m.group(1)
                i += 1
                continue
            m = _AUTOSUM_RE.match(line)
            if m:
                indent = len(m.group(1))
                i += 1
                while i < len(lines):
                    ln = lines[i]
                    if ln.strip() == "":
                        i += 1
                        continue
                    cur_indent = len(ln) - len(ln.lstrip())
                    if cur_indent <= indent:
                        break
                    if not _OPTION_RE.match(ln):
                        name = ln.strip()
                        if name and not name.startswith(".."):
                            entries.append((cur, name))
                    i += 1
                continue
            i += 1
    return entries


def _chain(root, parts):
    obj = root
    for p in parts:
        try:
            obj = getattr(obj, p)
        except Exception:
            return None
    return obj


def _robust_resolve(modname: str | None, dotted: str):
    """Resolve a name relative to `modname`, or as a fully-qualified path."""
    dotted = dotted.lstrip("~")
    candidates = [f"{modname}.{dotted}" if modname else dotted, dotted]
    for full in candidates:
        parts = full.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if not cand.startswith("statsmodels"):
                continue
            try:
                mod = importlib.import_module(cand)
            except ModuleNotFoundError:
                continue
            obj = _chain(mod, parts[i:])
            if obj is not None:
                return obj
    return None


def _srcinfo(obj):
    try:
        f = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
        return f, start
    except Exception:
        return None, None


def build_api_surface() -> list[dict]:
    """Every public function/class/method reachable from api.py exports or
    docs, resolved to (file, def-line) with dotted labels."""
    import functools

    records: dict[tuple[str, int], dict] = {}
    classes_seen: dict[type, str] = {}

    def add(obj, label):
        if isinstance(obj, functools.cached_property):
            obj = obj.func
        if isinstance(obj, property):
            obj = obj.fget
        if obj is None:
            return None
        if not (inspect.isfunction(obj) or inspect.isclass(obj)
                or inspect.ismethod(obj)):
            return None
        f, s = _srcinfo(obj)
        if not f or "statsmodels" not in f:
            return None
        try:
            rel = pathlib.Path(f).resolve().relative_to(
                PKG_ROOT.parent.resolve()).as_posix()
        except Exception:
            return None
        if "/sandbox/" in rel:
            # Excluded regardless of discovery path (also skipped in
            # coverage_public_api.cfg and class_coverage_report.py):
            # sandbox is explicitly documented as unfinished, uneven-quality
            # incubator code, not held to the same coverage bar.
            return None
        key = (rel, s)
        rec = records.setdefault(key, {"file": rel, "start": s, "labels": set()})
        rec["labels"].add(label)
        return obj

    for modname in _find_api_modules():
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            continue
        names = getattr(mod, "__all__", None) or [
            n for n in dir(mod) if not n.startswith("_")
        ]
        for n in names:
            obj = getattr(mod, n, None)
            if inspect.ismodule(obj):
                continue
            o = add(obj, f"{modname}.{n}")
            if o is not None and inspect.isclass(o):
                classes_seen.setdefault(o, f"{modname}.{n}")

    for modname, name in _parse_documented_names():
        if not modname:
            continue
        obj = _robust_resolve(modname, name)
        o = add(obj, f"{modname}.{name}")
        if o is not None and inspect.isclass(o):
            classes_seen.setdefault(o, f"{modname}.{name}")

    for cls, label in list(classes_seen.items()):
        for attr in dir(cls):
            if attr.startswith("_"):
                continue
            try:
                raw = inspect.getattr_static(cls, attr)
            except AttributeError:
                continue
            add(raw, f"{label}.{attr}")

    return [
        {"file": r["file"], "start": r["start"], "labels": sorted(r["labels"])}
        for r in records.values()
    ]


# --------------------------------------------------------------------------
# 2. Join against coverage data, measuring body (not def-line) coverage.
# --------------------------------------------------------------------------

_node_cache: dict[str, dict[int, tuple]] = {}


def _nodes_for(rel: str) -> dict[int, tuple]:
    if rel in _node_cache:
        return _node_cache[rel]
    m = {}
    fp = ROOT / rel
    try:
        tree = ast.parse(fp.read_text(encoding="utf8", errors="replace"))
    except Exception:
        _node_cache[rel] = m
        return m
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body_start = node.body[0].lineno if node.body else node.lineno
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                body_start = (node.body[1].lineno if len(node.body) > 1
                              else node.end_lineno + 1)
            rec = (body_start, node.end_lineno, node.name, type(node).__name__)
            m[node.lineno] = rec
            for d in getattr(node, "decorator_list", []):
                m[d.lineno] = rec
    _node_cache[rel] = m
    return m


def _is_abstract_stub(rel: str, start: int) -> bool:
    """True if the function body is exactly `raise NotImplementedError`
    or `pass` -- these are intentional and not coverage gaps."""
    node = None
    tree = ast.parse((ROOT / rel).read_text(encoding="utf8", errors="replace"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            starts = [n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]
            if start in starts:
                node = n
                break
    if node is None:
        return False
    stmts = [s for s in node.body
             if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                     and isinstance(s.value.value, str))]
    if len(stmts) == 1:
        s = stmts[0]
        if isinstance(s, ast.Pass):
            return True
        if isinstance(s, ast.Raise) and "NotImplementedError" in ast.dump(s):
            return True
    return False


def measure_gaps(api_surface: list[dict], coverage_json: dict) -> list[dict]:
    covmap = {}
    for k, v in coverage_json["files"].items():
        p = pathlib.Path(k)
        if not p.is_absolute():
            p = ROOT / p
        try:
            rel = p.resolve().relative_to(ROOT.resolve()).as_posix()
        except Exception:
            rel = p.as_posix()
        covmap[rel] = {
            "ex": set(v.get("executed_lines", [])),
            "mi": set(v.get("missing_lines", [])),
        }

    gaps = []
    for r in api_surface:
        rel, start = r["file"], r["start"]
        cm = covmap.get(rel)
        if cm is None:
            continue
        node = _nodes_for(rel).get(start)
        if node is None:
            continue
        body_start, end, _name, _ntype = node
        known = [ln for ln in range(body_start, end + 1)
                 if ln in cm["ex"] or ln in cm["mi"]]
        if not known:
            continue
        executed = sum(1 for ln in known if ln in cm["ex"])
        if executed > 0:
            continue
        if _is_abstract_stub(rel, start):
            continue
        gaps.append({
            "file": rel, "start": start,
            "label": min(r["labels"], key=len),
            "n_stmts": len(known),
        })
    gaps.sort(key=lambda g: (g["file"], g["start"]))
    return gaps


# --------------------------------------------------------------------------
# 3. CLI
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_json", help="path to a coverage.json file")
    parser.add_argument("--baseline", default=None,
                        help="baseline gap list; fail only on NEW gaps")
    parser.add_argument("--write-baseline", default=None,
                        help="write the current gap list to this path")
    args = parser.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        api_surface = build_api_surface()

    cov = json.loads(pathlib.Path(args.coverage_json).read_text(encoding="utf8"))
    gaps = measure_gaps(api_surface, cov)
    gap_keys = {f"{g['file']}:{g['start']}" for g in gaps}

    if args.write_baseline:
        pathlib.Path(args.write_baseline).write_text(
            json.dumps(sorted(gap_keys), indent=2), encoding="utf8")
        print(f"Wrote {len(gap_keys)} gaps to {args.write_baseline}")
        return 0

    if args.baseline:
        baseline = set(json.loads(
            pathlib.Path(args.baseline).read_text(encoding="utf8")))
        new_gaps = [g for g in gaps
                    if f"{g['file']}:{g['start']}" not in baseline]
        if new_gaps:
            print(f"{len(new_gaps)} NEW public API object(s) with zero test "
                  f"coverage (not present in {args.baseline}):\n")
            for g in new_gaps:
                print(f"  {g['label']:<64} {g['n_stmts']:>4} stmts  "
                      f"{g['file']}:{g['start']}")
            return 1
        print(f"No new zero-coverage public API objects "
              f"({len(gaps)} pre-existing, unchanged).")
        return 0

    print(f"{len(gaps)} public API object(s) with zero test coverage:\n")
    for g in gaps:
        print(f"  {g['label']:<64} {g['n_stmts']:>4} stmts  "
              f"{g['file']}:{g['start']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
