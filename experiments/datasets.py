"""Dataset registry: where each analysis host's data, question, human
comparison set, and plausibility reference live. Adding a dataset is one
entry here -- nothing else in the experiment code names a dataset."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from experiments import CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER = REPO_ROOT / "research/scratch/manyecoevo/extracted/master_data.csv"


def get(ds_id: str) -> dict:
    return {"id": ds_id, **CONFIG["datasets"][ds_id]}


def crowd(ds_id: str) -> list[dict]:
    """Human analyst estimates for the forest overlay: one row per analysis
    with its own sampling spread.
    REPORT-ONLY -- these numbers never enter a prompt."""
    d = get(ds_id)
    if d["crowd"] == "marp":
        # Match the publication's RQ1 plot: only rows the authors classified
        # as standardized regression betas. The table has no common sampling
        # SE field for these heterogeneous analyses, so the overlay is points.
        out = []
        src = REPO_ROOT / "research/scratch/marp/MARP_data_main.csv"
        for t in csv.DictReader(src.open(errors="replace")):
            if t.get("MARP_ES_type_1") != "beta":
                continue
            try:
                es = float(t["Standardized_ES_1"])
            except (TypeError, ValueError, KeyError):
                continue
            if abs(es) >= 1:
                continue
            out.append({"estimate": es, "se": 0})
        return out
    if not d["crowd"] or not MASTER.exists():
        return []
    rows = []
    for row in csv.DictReader(MASTER.open(errors="replace")):
        if (
            row.get("dataset") != d["crowd"]
            or row.get("exclusions_all") != "retain"
            or row.get("exclusions_effect_analysis") not in {"retain", "exclude_partial"}
        ):
            continue
        try:
            be, se, df = (
                float(row["beta_estimate"]),
                float(row["beta_SE"]),
                float(row["adjusted_df"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
        if se <= 0 or df <= 0:
            continue
        t = be / se
        r = t / math.sqrt(t * t + df)
        if abs(r) >= 1:
            continue
        # Match ManyEcoEvo::est_to_zr(): V(Zr) = 1 / adjusted_df.
        rows.append({"estimate": math.atanh(r), "se": 1 / math.sqrt(df)})
    return rows
