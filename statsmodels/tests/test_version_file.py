import importlib.util
import os
from pathlib import Path

import pytest

import statsmodels

GIT_VERSION = Path(statsmodels.__file__).parent / "_build" / "git_version.py"


@pytest.fixture(scope="module")
def git_version():
    pytest.importorskip("setuptools_scm")
    if not GIT_VERSION.exists():
        pytest.skip("The build scripts are not installed")
    spec = importlib.util.spec_from_file_location("sm_git_version", GIT_VERSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_version_file(git_version, tmp_path):
    # The editable install rewrites the version file whenever a process imports
    # statsmodels, while other processes may import it at the same time. The
    # file must never be truncated: an unchanged file is not written again and
    # a changed one is replaced in one step.
    target = tmp_path / "_version.py"
    git_version.write_version_file(str(target), "1.2.3", (1, 2, 3))
    assert "__version__ = version = '1.2.3'" in target.read_text()

    old = 1_000_000_000
    os.utime(target, (old, old))
    git_version.write_version_file(str(target), "1.2.3", (1, 2, 3))
    assert target.stat().st_mtime == old

    file_id = target.stat().st_ino
    git_version.write_version_file(str(target), "1.2.4", (1, 2, 4))
    assert "__version__ = version = '1.2.4'" in target.read_text()
    assert target.stat().st_ino != file_id
    assert [p.name for p in tmp_path.iterdir()] == ["_version.py"]
