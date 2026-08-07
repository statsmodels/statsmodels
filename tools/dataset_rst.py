#!/usr/bin/env python3
"""
Run this script to convert dataset documentation to ReST files. Relies
on the meta-information from the datasets of the currently installed version.
Ie., it imports the datasets package to scrape the meta-information.
"""
import inspect
from pathlib import Path
from string import Template

import statsmodels.api as sm

file_path = Path(__file__).resolve().parent
dest_dir = (file_path / ".." / "docs" / "source" / "datasets" / "generated").resolve()

datasets = dict(inspect.getmembers(sm.datasets, inspect.ismodule))
datasets.pop("utils")
last_mod_time = {}
for dataset, dataset_file in datasets.items():
    root = Path(dataset_file.__file__).resolve().parent
    files = list(root.glob("*"))
    if not files:
        raise NotImplementedError("Must be files to read the date")
    mtime = 0.0
    for f in files:
        if str(f).startswith("__") and f != "__init__.py":
            continue
        mtime = max(mtime, f.stat().st_mtime)
    last_mod_time[dataset] = mtime

doc_template = Template("""$TITLE
$title_

Description
-----------

$DESCRIPTION

Notes
-----
$NOTES

Source
------
$SOURCE

Copyright
---------

$COPYRIGHT\
""")

if __name__ == "__main__":

    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)

    for dataset, data_mod in datasets.items():
        rst_file_name = dataset + ".rst"
        write_pth = dest_dir / rst_file_name
        if write_pth.exists():
            rst_mtime = write_pth.stat().st_mtime
            if rst_mtime > last_mod_time[dataset]:
                print(
                    f"Skipping creation of {rst_file_name} since the rst file is newer "
                    "than the data files."
                )
                continue
        title = data_mod.TITLE
        descr = data_mod.DESCRLONG
        copyr = data_mod.COPYRIGHT
        notes = data_mod.NOTE
        source = data_mod.SOURCE
        write_file = doc_template.substitute(
            TITLE=title,
            title_="=" * len(title),
            DESCRIPTION=descr,
            NOTES=notes,
            SOURCE=source,
            COPYRIGHT=copyr,
        )
        print(f"Writing {rst_file_name}.")
        with write_pth.resolve().open("w", encoding="utf-8") as rst_file:
            rst_file.write(write_file)
