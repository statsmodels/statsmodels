"""Loaders for causal-inference benchmark datasets (Yao et al., TKDD 2021, Table 2)."""

from __future__ import annotations

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

IHDP_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/"
    "datasets/IHDP/csv/ihdp_npci_{replication}.csv"
)
IHDP_COLUMNS = (
    ["A", "y", "y_cf", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
)


def load_ihdp(
    replication: int = 1, cache_dir: str | Path = "data"
) -> pd.DataFrame:
    """Load one IHDP (NPCI setting) replication: 747 rows, 25 covariates.

    Columns: ``A`` (treatment), ``y`` (factual outcome), ``y_cf``
    (counterfactual outcome), ``mu0``/``mu1`` (noiseless potential
    outcomes), ``x1``..``x25`` (covariates). The file is downloaded once
    and cached in ``cache_dir``.
    """
    if not 1 <= replication <= 10:
        raise ValueError("CEVAE ships IHDP replications 1..10")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"ihdp_npci_{replication}.csv"
    if not path.exists():
        urllib.request.urlretrieve(IHDP_URL.format(replication=replication), path)
    frame = pd.read_csv(path, header=None)
    frame = frame.iloc[:, : len(IHDP_COLUMNS)]
    frame.columns = IHDP_COLUMNS
    frame["A"] = frame["A"].astype(float)
    return frame


def resample_rows(
    frame: pd.DataFrame, n_rows: int, seed: int = 0
) -> pd.DataFrame:
    """Bootstrap-resample rows to ``n_rows`` to scale a small dataset.

    Scaling is for systems measurements only; resampled rows add no new
    statistical information.
    """
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(frame), size=n_rows)
    return frame.iloc[index].reset_index(drop=True)
