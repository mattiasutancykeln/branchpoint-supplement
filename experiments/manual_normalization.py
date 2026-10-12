"""Executable row-level recipes for source-article plot normalization.

Dataset modules contain the evidence and numeric inputs for every accepted
result.  This module owns the formulas, validates their domains, and renders
the legacy ``standardization_overrides.json`` artifacts consumed by Layer 2.
No recipe may contain a precomputed value without the inputs that reproduce it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FISHER_FORMULA = (
    "q=estimate/se; r=q/sqrt(q**2+adjusted_df); "
    "value=atanh(r); se=1/sqrt(adjusted_df)"
)
BETA_FORMULA = (
    "scale=sd_x/sd_y; value=estimate*scale; se=reported_se*scale"
)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def materialize(recipe: dict) -> dict:
    """Calculate one override from its auditable inputs."""
    method = recipe["method"]
    inputs = recipe.get("inputs", {})
    evidence = recipe.get("evidence")
    if not evidence:
        raise ValueError("normalization recipe lacks evidence")

    if method == "coefficient_se_adjusted_df_to_fisher_z":
        estimate = inputs.get("reported_estimate")
        reported_se = inputs.get("reported_se")
        degrees = inputs.get("adjusted_df")
        df_method = inputs.get("df_method")
        if not all(_finite(value) for value in (estimate, reported_se, degrees)):
            raise ValueError("Fisher-z recipe has non-finite inputs")
        if not isinstance(df_method, str) or not df_method.strip():
            raise ValueError("Fisher-z recipe lacks an evidenced finite-df method")
        if "benchmark" in df_method.lower():
            raise ValueError("Fisher-z recipe may not construct df from a benchmark convention")
        if reported_se <= 0 or degrees <= 1:
            raise ValueError("Fisher-z recipe requires SE > 0 and adjusted_df > 1")
        q = estimate / reported_se
        correlation = q / math.sqrt(q * q + degrees)
        value = math.atanh(correlation)
        standard_error = 1 / math.sqrt(degrees)
        metric = "fisher_z"
        formula = FISHER_FORMULA
    elif method == "direct_fisher_z":
        value, standard_error = inputs.get("value"), inputs.get("se")
        if not _finite(value) or (standard_error is not None and not _finite(standard_error)):
            raise ValueError("direct Fisher-z recipe has non-finite inputs")
        metric, formula = "fisher_z", "value and se reported directly on Fisher-z scale"
    elif method == "direct_r_to_fisher_z":
        correlation, se_r = inputs.get("r"), inputs.get("se_r")
        if not _finite(correlation) or abs(correlation) >= 1:
            raise ValueError("direct-r recipe requires finite abs(r) < 1")
        value = math.atanh(correlation)
        standard_error = None if se_r is None else se_r / (1 - correlation * correlation)
        metric, formula = "fisher_z", "value=atanh(r); se=se_r/(1-r**2)"
    elif method == "fitted_linear_rescaling_to_standardized_beta":
        estimate = inputs.get("reported_estimate")
        reported_se = inputs.get("reported_se")
        sd_x, sd_y = inputs.get("sd_x"), inputs.get("sd_y")
        if not all(_finite(value) for value in (estimate, reported_se, sd_x, sd_y)):
            raise ValueError("standardized-beta recipe has non-finite inputs")
        if reported_se < 0 or sd_x <= 0 or sd_y <= 0:
            raise ValueError("standardized-beta recipe has invalid scale inputs")
        scale = sd_x / sd_y
        value, standard_error = estimate * scale, reported_se * scale
        metric, formula = "standardized_beta", BETA_FORMULA
    elif method == "direct_standardized_beta":
        value, standard_error = inputs.get("value"), inputs.get("se")
        if not _finite(value) or (standard_error is not None and not _finite(standard_error)):
            raise ValueError("direct standardized-beta recipe has non-finite inputs")
        metric, formula = "standardized_beta", "value and se reported as fully standardized beta"
    elif method == "excluded":
        reason = recipe.get("reason")
        if not reason:
            raise ValueError("excluded recipe lacks a reason")
        return {
            "metric": recipe["metric"],
            "value": None,
            "se": None,
            "method": "excluded",
            "reason": reason,
            "inputs": inputs,
            "evidence": evidence,
            "formula": None,
            "primary": False,
        }
    else:
        raise ValueError(f"unknown normalization method {method!r}")

    return {
        "metric": metric,
        "value": value,
        "se": standard_error,
        "method": method,
        "inputs": inputs,
        "evidence": evidence,
        "formula": formula,
        "primary": True,
    }


def row_ids(condition: Path) -> list[str]:
    """Return Layer-2 row identifiers in source order."""
    rows = json.loads(condition.joinpath("results.json").read_text())["rows"]
    result = []
    for row in rows:
        if "seed" in row:
            result.append(f"seed{row['seed']:03d}")
        else:
            result.append(row["branch"])
    return result


def validate_condition(condition: Path, recipes: dict[str, dict]) -> dict[str, dict]:
    """Require exactly one reproducible disposition for every result row."""
    source = json.loads(condition.joinpath("results.json").read_text())
    expected = row_ids(condition)
    missing = [row_id for row_id in expected if row_id not in recipes]
    extra = [row_id for row_id in recipes if row_id not in expected]
    if missing or extra:
        raise ValueError(f"{condition}: missing={missing}, extra={extra}")
    expected_metric = "standardized_beta" if source["dataset"] == "marp" else "fisher_z"
    overrides = {}
    for row_id, row in zip(expected, source["rows"], strict=True):
        recipe = recipes[row_id]
        inputs = recipe.get("inputs", {})
        for input_key, row_key in (
            ("reported_estimate", "estimate"),
            ("reported_se", "se"),
        ):
            recorded, reported = inputs.get(input_key), row.get(row_key)
            if recorded is not None and reported is not None and not math.isclose(
                recorded, reported, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(
                    f"{condition}/{row_id}: {input_key} does not match results.json"
                )
        override = materialize(recipe)
        if override["metric"] != expected_metric:
            raise ValueError(f"{condition}/{row_id}: wrong target metric")
        overrides[row_id] = override
    return overrides


def write_condition(condition: Path, recipes: dict[str, dict]) -> Path:
    """Write the derived compatibility artifact after complete validation."""
    overrides = validate_condition(condition, recipes)
    path = condition / "standardization_overrides.json"
    path.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    return path
