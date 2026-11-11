"""Rank estimation classes by untested code, from a coverage.json.

Complements ``tools/check_public_api_coverage.py``.  That script answers
"which public objects are *never* executed"; this one answers "which
estimation *classes* -- models, their results classes, and the
intermediate machinery they lean on -- carry the most untested code, and
which of their methods are the gaps".

It is deliberately pure-AST: it builds the class inheritance graph
textually across the whole package, so it runs without importing (and
therefore without building) statsmodels.

Classification
--------------
model
    Transitively derives from ``Model`` / ``LikelihoodModel``.
results
    Transitively derives from ``Results`` / ``LikelihoodModelResults``,
    or is named ``*Results`` / ``*Result``.
wrapper
    A ``ResultsWrapper`` subclass.
support
    Everything else -- covariance structures, families, smoothers,
    initialization, margins, influence, ... i.e. the intermediate
    classes that form part of estimation.

Methods whose body is exactly ``pass`` or ``raise NotImplementedError``
are abstract and are excluded: they are not coverage gaps.

Usage
-----
Produce a coverage.json from a full run::

    pytest statsmodels -n auto -m "not slow and not example" \\
        --cov=statsmodels --cov-config=tools/coverage_public_api.cfg \\
        --cov-report=json:coverage.json

then::

    python tools/class_coverage_report.py coverage.json
    python tools/class_coverage_report.py coverage.json --kind model
    python tools/class_coverage_report.py coverage.json --dead
    python tools/class_coverage_report.py coverage.json --by-package
    python tools/class_coverage_report.py coverage.json --json out.json
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "statsmodels"
SKIP_PARTS = ("tests", "sandbox", "examples")

MODEL_ROOTS = {"Model", "LikelihoodModel", "GenericLikelihoodModel"}
RESULT_ROOTS = {"Results", "LikelihoodModelResults", "ResultMixin"}
WRAP_ROOTS = {"ResultsWrapper", "ResultsWrapperMixin"}


class ClassInfo:
    __slots__ = ("bases", "file", "lineno", "methods", "name")

    def __init__(self, name, file, lineno, bases, methods):
        self.name = name
        self.file = file
        self.lineno = lineno
        self.bases = bases
        self.methods = methods  # (name, body_start, end_line, kind)


def _base_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, (ast.Subscript, ast.Call)):
        return _base_name(node.value if isinstance(node, ast.Subscript)
                          else node.func)
    return ""


def _body_start(node) -> int:
    """First line of the body, skipping the docstring."""
    if not node.body:
        return node.lineno
    first = node.body[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return (node.body[1].lineno if len(node.body) > 1
                else node.end_lineno + 1)
    return first.lineno


def _is_stub(node) -> bool:
    """Body is exactly ``pass`` or ``raise NotImplementedError``."""
    stmts = [s for s in node.body
             if not (isinstance(s, ast.Expr)
                     and isinstance(s.value, ast.Constant)
                     and isinstance(s.value.value, str))]
    if len(stmts) != 1:
        return not stmts
    stmt = stmts[0]
    if isinstance(stmt, ast.Pass):
        return True
    return (isinstance(stmt, ast.Raise)
            and "NotImplementedError" in ast.dump(stmt))


def _method_kind(node) -> str:
    decs = {_base_name(d) for d in node.decorator_list}
    if {"property", "cached_property", "cache_readonly"} & decs:
        return "property"
    if "staticmethod" in decs:
        return "static"
    if "classmethod" in decs:
        return "class"
    return "method"


def collect_classes() -> list[ClassInfo]:
    classes = []
    for path in sorted(PKG.rglob("*.py")):
        if any(p in SKIP_PARTS for p in path.parts):
            continue
        if path.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.resolve().relative_to(ROOT).as_posix()
        stack = list(tree.body)
        while stack:
            node = stack.pop()
            if not isinstance(node, ast.ClassDef):
                continue
            methods = []
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_stub(member):
                        continue
                    methods.append((member.name, _body_start(member),
                                    member.end_lineno, _method_kind(member)))
                elif isinstance(member, ast.ClassDef):
                    stack.append(member)
            classes.append(ClassInfo(node.name, rel, node.lineno,
                                     [_base_name(b) for b in node.bases],
                                     methods))
    return classes


def classify(cls: ClassInfo, by_name: dict[str, list[ClassInfo]]) -> str:
    seen: set[str] = set()
    anc = {cls.name}
    todo = list(cls.bases)
    while todo:
        base = todo.pop()
        if not base or base in seen:
            continue
        seen.add(base)
        anc.add(base)
        for parent in by_name.get(base, []):
            todo.extend(parent.bases)
    if anc & WRAP_ROOTS:
        return "wrapper"
    if anc & RESULT_ROOTS or cls.name.endswith(("Results", "Result")):
        return "results"
    if anc & MODEL_ROOTS:
        return "model"
    if cls.name.endswith("Wrapper"):
        return "wrapper"
    return "support"


def load_coverage(path: pathlib.Path) -> dict[str, tuple[set, set]]:
    data = json.loads(path.read_text(encoding="utf8"))
    out = {}
    for key, val in data["files"].items():
        p = pathlib.Path(key)
        p = p if p.is_absolute() else ROOT / p
        try:
            rel = p.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = p.as_posix()
        out[rel] = (set(val.get("executed_lines", [])),
                    set(val.get("missing_lines", [])))
    return out


def build_rows(covmap) -> list[dict]:
    classes = collect_classes()
    by_name: dict[str, list[ClassInfo]] = defaultdict(list)
    for cls in classes:
        by_name[cls.name].append(cls)

    rows = []
    for cls in classes:
        cov = covmap.get(cls.file)
        if cov is None:
            continue
        executed, missing = cov
        total = hit = 0
        dead, weak = [], []
        for name, start, end, kind in cls.methods:
            known = [ln for ln in range(start, end + 1)
                     if ln in executed or ln in missing]
            if not known:
                continue
            run = sum(1 for ln in known if ln in executed)
            total += len(known)
            hit += run
            if run == 0:
                dead.append((name, len(known), kind))
            elif len(known) - run >= 3 and run / len(known) < 0.7:
                weak.append((name, len(known) - run, len(known), kind))
        if total == 0:
            continue
        rows.append({
            "cls": cls.name, "file": cls.file, "line": cls.lineno,
            "kind": classify(cls, by_name),
            "total": total, "hit": hit, "missing": total - hit,
            "pct": 100.0 * hit / total,
            "dead": sorted(dead, key=lambda t: -t[1]),
            "weak": sorted(weak, key=lambda t: -t[1]),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_json", help="path to a coverage.json file")
    parser.add_argument("--kind", action="append",
                        choices=["model", "results", "wrapper", "support"],
                        help="restrict to these class kinds (repeatable)")
    parser.add_argument("--dead", action="store_true",
                        help="list zero-coverage methods instead of classes")
    parser.add_argument("--by-package", action="store_true",
                        help="aggregate by subpackage instead of by class")
    parser.add_argument("--min-missing", type=int, default=1,
                        help="hide entries below this many missing statements")
    parser.add_argument("--limit", type=int, default=40,
                        help="maximum rows to print (0 for all)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also dump the full result set to this path")
    args = parser.parse_args(argv)

    covmap = load_coverage(pathlib.Path(args.coverage_json))
    rows = build_rows(covmap)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(rows, indent=1), encoding="utf8")

    kinds = set(args.kind) if args.kind else {"model", "results", "wrapper",
                                              "support"}
    sel = [r for r in rows if r["kind"] in kinds]

    print(f"{len(rows)} classes analysed from {args.coverage_json}")
    for kind in ("model", "results", "wrapper", "support"):
        sub = [r for r in rows if r["kind"] == kind]
        tot = sum(r["total"] for r in sub)
        hit = sum(r["hit"] for r in sub)
        print(f"  {kind:<8} {len(sub):>4} classes  {hit:>6}/{tot:<6} stmts  "
              f"{100.0 * hit / max(tot, 1):5.1f}%")
    print()

    if args.by_package:
        agg = defaultdict(lambda: {"missing": 0, "total": 0, "dead": 0,
                                   "n": 0})
        for r in sel:
            parts = r["file"].split("/")
            pkg = "/".join(parts[1:3]) if len(parts) > 3 else parts[1]
            rec = agg[pkg]
            rec["missing"] += r["missing"]
            rec["total"] += r["total"]
            rec["dead"] += sum(d[1] for d in r["dead"])
            rec["n"] += 1
        print(f'{"miss":>6} {"dead":>5} {"total":>6} {"pct":>6} {"cls":>4}'
              f'  subpackage')
        for pkg, rec in sorted(agg.items(), key=lambda kv: -kv[1]["missing"]):
            pct = 100.0 * (rec["total"] - rec["missing"]) / rec["total"]
            print(f'{rec["missing"]:6d} {rec["dead"]:5d} {rec["total"]:6d} '
                  f'{pct:5.1f}% {rec["n"]:4d}  {pkg}')
        return 0

    if args.dead:
        sub = [r for r in sel if r["dead"]]
        sub.sort(key=lambda r: -sum(d[1] for d in r["dead"]))
        grand = sum(sum(d[1] for d in r["dead"]) for r in sub)
        print(f"{grand} zero-coverage statements across {len(sub)} classes\n")
        shown = sub if args.limit == 0 else sub[:args.limit]
        for r in shown:
            n = sum(d[1] for d in r["dead"])
            if n < args.min_missing:
                continue
            names = ", ".join(f"{d[0]}({d[1]})" for d in r["dead"][:7])
            print(f'{n:5d}  {r["cls"]:<32} {r["file"]}:{r["line"]}\n'
                  f'       {names}')
        return 0

    sub = [r for r in sel if r["missing"] >= args.min_missing]
    sub.sort(key=lambda r: -r["missing"])
    shown = sub if args.limit == 0 else sub[:args.limit]
    print(f'{"miss":>5} {"pct":>6} {"total":>6}  class')
    for r in shown:
        print(f'{r["missing"]:5d} {r["pct"]:5.1f}% {r["total"]:6d}  '
              f'{r["cls"]:<30} {r["file"]}:{r["line"]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
