"""Banked post-processing: extract → standardize → scope → result.

    python -m experiments.layer2 results/grid2/bt_sol_branchpoint

The sampler explores. This module makes every arm comparable on the declared
estimand. Each stage writes its evidence before the next stage can consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import branchpoint as bp
from experiments import CONFIG, write_json
from experiments.manual_normalization import materialize as materialize_normalization

REPO_ROOT = Path(__file__).resolve().parents[1]
Judge = Callable[[str, dict], dict]
SCHEMAS = json.loads(Path(__file__).with_name("layer2.schemas.json").read_text())
STAT_SCHEMA, SCOPE_SCHEMA = SCHEMAS["stat"], SCHEMAS["scope"]
META = ("dataset", "mode", "model", "allocation", "seed", "judge_model", "usd")
SCALE_EVIDENCE = "STANDARDIZATION_EVIDENCE "


def _digest(prompt: str, schema: dict, judge: Judge) -> str:
    payload = [prompt, schema, getattr(judge, "model", None)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _judge(judge: Judge, prompt: str, schema: dict, **linkage) -> dict:
    call = getattr(judge, "call", None)
    return call(prompt, schema, **linkage) if call else judge(prompt, schema)


def _judge_missing(
    prompts: dict[str, str],
    judge: Judge,
    schema: dict,
    workers: int,
    *,
    stage: str,
    condition: str,
) -> dict:
    keys = list(prompts)

    def run(key: str) -> dict:
        return _judge(
            judge,
            prompts[key],
            schema,
            stage=stage,
            condition=condition,
            segment=key,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(keys, pool.map(run, keys), strict=True))


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def _attempt(segment: str) -> tuple:
    match = re.search(r"[-_](\d+)$", segment)
    return (match is None, int(match.group(1)) if match else segment)


def branches(project: Path) -> list[dict]:
    """Read the root and novelty-reviewed branches retained by the sampler."""
    state_path = project / bp.STATE
    state = json.loads(state_path.read_text()) if state_path.exists() else {"registry": {}}
    segments = [
        "",
        *sorted(
            (
                segment
                for decision in state.get("registry", {}).values()
                for segment in decision.get("branches", [])
            ),
            key=_attempt,
        ),
    ]
    reports = []
    for segment in segments:
        report = bp.Project(project, segment).report()
        if report and report.get("estimate") is not None:
            reports.append({"branch": Path(segment).name if segment else "root", **report})
    return reports


def _clean(note: str | None) -> str:
    return note or ""


def _sections(question: str, *names: str) -> str:
    def section(name: str) -> str:
        match = re.search(rf"##\s*{name}\s*\n(.*?)(?=\n##\s|\Z)", question, re.S)
        return match.group(1).strip() if match else ""

    return "\n".join(f"{name}: {text}" for name in names if (text := section(name)))


def _printed(project: bp.Project, limit: int = 6000) -> str:
    output = []
    for event in project.events():
        if event["kind"] != "code":
            continue
        try:
            stdout = json.loads(event["content"]).get("stdout") or ""
        except (json.JSONDecodeError, TypeError):
            continue
        if stdout:
            output.append(stdout)
    return "\n".join(output)[-limit:]


def _runs(condition: Path) -> list[tuple[dict, bp.Project]]:
    rows = json.loads((condition / "results.json").read_text())["rows"]
    runs = []
    for row in rows:
        if "seed" in row:
            run_id = f"seed{row['seed']:03d}"
            project = bp.Project(condition / run_id)
        else:
            run_id = row["branch"]
            segment = "" if run_id == "root" else f"{bp.FORKS}/{run_id}"
            project = bp.Project(condition, segment)
        runs.append(({**row, "id": run_id}, project))
    return runs


def extract(condition: Path, judge: Judge, workers: int = 1) -> dict:
    """Read focal statistics from each run's own output; bank once."""
    path = condition / "stats.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old_stats = old.get("stats", {})
    old_digests = old.get("input_digests", {})
    stats, digests, prompts = {}, {}, {}
    for row, project in _runs(condition):
        if row.get("estimate") is None:
            continue
        prompt = CONFIG["prompts"]["extract"].format(
            printed=_printed(project), note=_clean(row.get("note"))
        )
        digest = _digest(prompt, STAT_SCHEMA, judge)
        digests[row["id"]] = digest
        if old_digests.get(row["id"]) == digest:
            stats[row["id"]] = old_stats[row["id"]]
        else:
            prompts[row["id"]] = prompt
    stats.update(
        _judge_missing(
            prompts,
            judge,
            STAT_SCHEMA,
            workers,
            stage="extract",
            condition=condition.name,
        )
    )
    result = {
        "condition": condition.name,
        "judge_model": getattr(judge, "model", None),
        "input_digests": digests,
        "stats": stats,
    }
    write_json(path, result)
    return result


def _df_for(stat: dict, df_policy: str | None, kind: str) -> tuple[float | None, str]:
    """The df to pair with a test statistic, on ONE scale per dataset.

    Fisher-z SE is 1/sqrt(df), so a row standardized on a cluster df and its
    neighbour standardized on a row df get intervals differing several-fold for
    the same analysis, and the gap between the two scales is read as spread.
    df_policy therefore decides which df WINS, not merely which fills a gap; it
    records the scale each dataset's own human comparator used.
    """
    reference, resid, n_obs = stat.get("df_reference"), stat.get("df_resid"), stat.get("n_obs")
    if df_policy == "row":
        # Row-scale comparator: a row-scale df wins over a reported cluster df,
        # otherwise neighbouring rows land on scales that differ several-fold.
        # A GLM reports an asymptotic Wald z and no classical error df; n_obs-2
        # is the conservative end and k is immaterial at these n.
        if resid is not None:
            return resid, f"{kind}_df_resid"
        if n_obs:
            return float(n_obs) - 2, f"{kind}_df_from_n_obs"
        if reference is not None:
            return reference, f"{kind}_df_reference"
        return None, f"{kind}_without_finite_df"
    # "cluster" is the default order already: a reported cluster df wins, and a
    # row-scale df is still better than dropping the row. Rows that carry only a
    # row-scale df stay on it -- that is a data limitation, not a precedence
    # choice, and it is reported as such rather than silently reconciled.
    if reference is not None:
        return reference, f"{kind}_df_reference"
    return resid, f"{kind}_df_resid"


def _standardized(
    stat: dict, reported_se, df_policy: str | None = None
) -> tuple[float | None, str, float | None, str]:
    """The sole owner of conversion to the study's final Fisher-z scale."""
    t, z = stat.get("t_stat"), stat.get("z_stat")
    pooled = stat.get("design") == "meta_analytic_pooling"
    blocked = None
    if t is not None and z is not None:
        blocked = "ambiguous_test_statistic"
    elif t is not None and not pooled:
        df, method = _df_for(stat, df_policy, "t")
        degrees = float(df) if df is not None else None
        if degrees is not None and math.isfinite(degrees) and degrees > 1:
            t = float(t)
            value = math.atanh(t / math.sqrt(t * t + degrees))
            se = 1 / math.sqrt(degrees)
            return value, method, se, method
        blocked = "t_without_finite_df"
    elif z is not None and not pooled:
        df, method = _df_for(stat, df_policy, "z")
        degrees = float(df) if df is not None else None
        if degrees is not None and math.isfinite(degrees) and degrees > 1:
            z = float(z)
            value = math.atanh(z / math.sqrt(z * z + degrees))
            se = 1 / math.sqrt(degrees)
            return value, method, se, method
        blocked = "asymptotic_z_without_finite_df"
    elif pooled:
        blocked = "pooled_without_direct_effect"
    effect, kind = stat.get("direct_effect"), stat.get("direct_effect_kind")
    if effect is None or kind == "none":
        method = blocked or "missing_test_statistic"
        return None, method, None, method
    effect = float(effect)
    if kind == "fisher_z":
        return effect, "as_reported_fisher_z", reported_se, "as_reported"
    if abs(effect) < 1:
        prefix = "pooled_" if pooled else "atanh_"
        return math.atanh(effect), prefix + kind, reported_se, "as_reported"
    return None, "unsupported_direct_effect", None, "unsupported_direct_effect"


def _spread(rows: list[dict]) -> dict:
    values = sorted(row["estimate_standardized"] for row in rows)
    if not values:
        return {
            "n": 0,
            "envelope": None,
            "width": None,
            "width_q05_q95": None,
            "median": None,
            "unique_ends": 0,
        }
    return {
        "n": len(values),
        "envelope": [values[0], values[-1]],
        "width": values[-1] - values[0] if len(values) > 1 else None,
        "width_q05_q95": (
            quantile(values, 0.95) - quantile(values, 0.05) if len(values) > 1 else None
        ),
        "median": quantile(values, 0.5),
        "unique_ends": len({round(value, 3) for value in values}),
    }


def _fitted_scale_override(row: dict, project: bp.Project) -> dict | None:
    """Read a matching fitted-scale record banked by a Layer-1 script."""
    for event in reversed(project.events()):
        if event["kind"] != "code":
            continue
        try:
            stdout = json.loads(event["content"]).get("stdout") or ""
        except (json.JSONDecodeError, TypeError):
            continue
        for line in reversed(stdout.splitlines()):
            if not line.startswith(SCALE_EVIDENCE):
                continue
            try:
                inputs = json.loads(line.removeprefix(SCALE_EVIDENCE))
            except json.JSONDecodeError as error:
                raise ValueError(f"{project.path}/trace.jsonl event {event['i']}: invalid scale evidence") from error
            if (inputs.get("reported_estimate"), inputs.get("reported_se")) != (
                row.get("estimate"),
                row.get("se"),
            ):
                continue
            return {
                "metric": "standardized_beta",
                "method": "exact_fitted_sample_linear_rescaling",
                "inputs": inputs,
                "evidence": f"{project.path}/trace.jsonl event {event['i']} code stdout",
            }
    return None


def _override_values(condition: Path, row: dict, override: dict, metric: str):
    """Reconstruct audited values when an override banks fitted-sample scales."""
    label = f"{condition.name}/{row['id']}"
    if override.get("metric") != metric:
        raise ValueError(f"{label}: override metric mismatch")
    if override.get("method") in {
        "coefficient_se_adjusted_df_to_fisher_z",
        "direct_fisher_z",
        "direct_r_to_fisher_z",
        "fitted_linear_rescaling_to_standardized_beta",
        "direct_standardized_beta",
    }:
        reconstructed = materialize_normalization(override)
        for key in ("value", "se"):
            recorded, calculated = override.get(key), reconstructed.get(key)
            if recorded is None or calculated is None:
                if recorded != calculated:
                    raise ValueError(f"{label}: recorded {key} does not match reconstruction")
            elif not math.isclose(recorded, calculated, rel_tol=1e-12):
                raise ValueError(f"{label}: recorded {key} does not match reconstruction")
        return reconstructed["value"], reconstructed.get("se")
    if override.get("method") not in {
        "exact_fitted_sample_linear_rescaling",
        "reported_fully_standardized_beta",
    }:
        return override["value"], override.get("se")
    inputs = override.get("inputs", {})
    if not override.get("evidence"):
        raise ValueError(f"{label}: missing standardization evidence")
    if (inputs.get("reported_estimate"), inputs.get("reported_se")) != (
        row.get("estimate"),
        row.get("se"),
    ):
        raise ValueError(f"{label}: standardization evidence does not match report")
    sd_x, sd_y = inputs.get("sd_x"), inputs.get("sd_y")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in (sd_x, sd_y)):
        raise ValueError(f"{label}: invalid fitted-sample scale")
    scale = sd_x / sd_y
    value = row["estimate"] * scale
    standard_error = row.get("se") * scale if row.get("se") is not None else None
    for key, reconstructed in (("value", value), ("se", standard_error)):
        recorded = override.get(key)
        if recorded is not None and not math.isclose(recorded, reconstructed, rel_tol=1e-12):
            raise ValueError(f"{label}: recorded {key} does not match reconstruction")
    return value, standard_error


def standardize(condition: Path) -> dict:
    """Put each dataset on its human comparator's metric using recorded rules."""
    source = json.loads((condition / "results.json").read_text())
    stats_path = condition / "stats.json"
    bank = json.loads(stats_path.read_text()) if stats_path.exists() else {"stats": {}}
    stats = bank["stats"]
    target_metric = "standardized_beta" if source["dataset"] == "marp" else "fisher_z"
    override_path = condition / "standardization_overrides.json"
    overrides = json.loads(override_path.read_text()) if override_path.exists() else {}
    rows, methods = [], {}
    for row, project in _runs(condition):
        stat = stats.get(row["id"], {})
        override = overrides.get(row["id"])
        if not override and target_metric == "standardized_beta":
            override = _fitted_scale_override(row, project)
        if override and not override.get("primary", True):
            value = standard_error = None
            method = se_method = override["method"]
            transformation = override
        elif override:
            value, standard_error = _override_values(condition, row, override, target_metric)
            method = se_method = override["method"]
            transformation = override
        elif row.get("estimate") is None:
            value, method, standard_error, se_method = None, "no_estimate", None, "no_estimate"
            transformation = {"metric": target_metric, "method": method, "inputs": {}}
        elif target_metric == "standardized_beta":
            direct, kind = stat.get("direct_effect"), stat.get("direct_effect_kind")
            if direct is not None and kind == "beta_std":
                value, standard_error = float(direct), row.get("se")
                method = se_method = "as_reported_beta_std"
            else:
                value = standard_error = None
                method = se_method = "requires_standardized_beta"
            transformation = {
                "metric": target_metric,
                "method": method,
                "inputs": {"direct_effect": direct, "direct_effect_kind": kind},
                "evidence": "stats.json and terminal Layer-1 report",
            }
        else:
            value, method, standard_error, se_method = _standardized(
                stat, row.get("se"), CONFIG["datasets"][source["dataset"]].get("df_policy")
            )
            transformation = {
                "metric": target_metric,
                "method": method,
                "inputs": {
                    "t_stat": stat.get("t_stat"),
                    "z_stat": stat.get("z_stat"),
                    "df_resid": stat.get("df_resid"),
                    "df_reference": stat.get("df_reference"),
                    "direct_effect": stat.get("direct_effect"),
                    "direct_effect_kind": stat.get("direct_effect_kind"),
                },
                "evidence": "stats.json and terminal Layer-1 report",
            }
        orientation_applied = value is not None and row.get("orientation") == "reversed"
        if orientation_applied:
            value = -value
        methods[method] = methods.get(method, 0) + 1
        rows.append(
            {
                **row,
                "estimate_as_reported": row["estimate"],
                "estimate_standardized": value,
                "standardized_via": method,
                "se_as_reported": row.get("se"),
                "se_standardized": standard_error,
                "se_standardized_via": se_method,
                "orientation_applied": orientation_applied,
                "target_metric": target_metric,
                "transformation": transformation,
                "stats": stat,
            }
        )
    usable = [row for row in rows if row["estimate_standardized"] is not None]
    result = {
        "condition": condition.name,
        **{key: source.get(key) for key in META},
        "layer2_model": bank.get("judge_model"),
        "target_metric": target_metric,
        **_spread(usable),
        "n_failed": methods.get("no_estimate", 0),
        "n_unstandardizable": sum(
            row.get("estimate") is not None and row["estimate_standardized"] is None
            for row in rows
        ),
        "via": methods,
        "rows": rows,
    }
    write_json(condition / "results_standardized.json", result)
    return result


def scope(condition: Path, judge: Judge, workers: int = 1) -> dict:
    """Judge only whether each standardized result answers the declared question."""
    path = condition / "scope.json"
    result = json.loads((condition / "results_standardized.json").read_text())
    question = (REPO_ROOT / CONFIG["datasets"][result["dataset"]]["question"]).read_text()
    frame = _sections(question, "Declared estimand", "Frame policy")
    old = json.loads(path.read_text()) if path.exists() else {}
    old_scopes = old.get("scopes", {})
    old_digests = old.get("input_digests", {})
    scopes, digests, prompts = {}, {}, {}
    for row in result["rows"]:
        if row["estimate_standardized"] is None:
            continue
        stat = row["stats"]
        prompt = CONFIG["prompts"]["scope"].format(
            frame=frame,
            predictor=row.get("predictor") or "not reported",
            orientation=row.get("orientation") or "not reported",
            note=_clean(row.get("note")),
            n_obs=stat.get("n_obs"),
            unit=stat.get("unit"),
        )
        digest = _digest(prompt, SCOPE_SCHEMA, judge)
        digests[row["id"]] = digest
        if old_digests.get(row["id"]) == digest:
            scopes[row["id"]] = old_scopes[row["id"]]
        else:
            prompts[row["id"]] = prompt
    scopes.update(
        _judge_missing(
            prompts,
            judge,
            SCOPE_SCHEMA,
            workers,
            stage="scope",
            condition=condition.name,
        )
    )
    banked = {
        "condition": condition.name,
        "dataset": result["dataset"],
        "judge_model": getattr(judge, "model", None),
        "input_digests": digests,
        "scopes": scopes,
    }
    write_json(path, banked)
    return banked


def finalize(condition: Path) -> dict:
    """Write the reportable spread; retain every excluded row and verdict."""
    source = json.loads((condition / "results_standardized.json").read_text())
    scopes = json.loads((condition / "scope.json").read_text())["scopes"]
    rows = [
        {
            **row,
            "scope": (
                scopes.get(row["id"], {}).get("scope", "U")
                if row["estimate_standardized"] is not None
                else None
            ),
        }
        for row in source["rows"]
    ]
    judged = [row for row in rows if row["scope"] is not None]
    live = [row for row in judged if row["scope"] == "E"]
    result = {
        **{key: source.get(key) for key in ("condition", *META, "layer2_model")},
        **_spread(live),
        "n_judged": len(judged),
        "n_failed": sum(row.get("standardized_via") == "no_estimate" for row in rows),
        "n_unstandardizable": sum(
            row.get("estimate_standardized") is None
            and row.get("standardized_via") != "no_estimate"
            for row in rows
        ),
        "out_of_scope": sum(row["scope"] == "N" for row in judged),
        "unclear": sum(row["scope"] == "U" for row in judged),
        "rows": rows,
    }
    write_json(condition / "results_final.json", result)
    return result


def condition_paths(
    root: str | None, names: list[str], only: str | None
) -> tuple[Path | None, list[Path]]:
    if root and names:
        raise ValueError("use --root or explicit condition directories, not both")
    root_path = None
    if root:
        root_path = Path(root)
        root_path = root_path if root_path.is_absolute() else REPO_ROOT / root_path
        names = [
            str(path) for path in sorted(root_path.iterdir()) if (path / "results.json").is_file()
        ]
    paths = [Path(name) if Path(name).is_absolute() else REPO_ROOT / name for name in names]
    paths = [path for path in paths if not only or only in path.name]
    if not paths:
        raise ValueError("no completed Layer-1 conditions selected")
    if missing := [path for path in paths if not (path / "results.json").is_file()]:
        raise ValueError(f"missing results.json: {', '.join(str(path) for path in missing)}")
    return root_path, paths


def judge_model(paths: list[Path], requested: str | None) -> str:
    if requested:
        return requested
    models = {
        json.loads(path.joinpath("results.json").read_text()).get("judge_model")
        for path in paths
    }
    models.discard(None)
    if len(models) != 1:
        raise ValueError("selected conditions do not declare one judge model; pass --model")
    return models.pop()


def update_status(root: Path, **fields) -> None:
    path = root / "run_status.json"
    try:
        status = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        status = {}
    write_json(path, {**status, **fields, "updated": time.time()})


def main() -> None:
    from experiments import executors

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "conditions", nargs="*", help="condition directories containing results.json"
    )
    parser.add_argument(
        "--root", help="discover every completed condition directly under this result root"
    )
    parser.add_argument("--only", help="condition-name substring")
    parser.add_argument("--model", help="default: judge_model saved by the selected conditions")
    parser.add_argument("--ledger", help="default: <root>/costs.jsonl, or configured ledger")
    parser.add_argument("--guard", type=float, default=500.0)
    parser.add_argument("--workers", type=int, default=6, help="concurrent Layer-2 judge calls")
    parser.add_argument(
        "--figures", action="store_true", help="render figures after Layer 2 (requires --root)"
    )
    args = parser.parse_args()
    try:
        root, conditions = condition_paths(args.root, args.conditions, args.only)
        model = judge_model(conditions, args.model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.figures and root is None:
        parser.error("--figures requires --root")
    if args.workers < 1:
        parser.error("--workers must be positive")
    ledger = (
        Path(args.ledger)
        if args.ledger
        else root / "costs.jsonl"
        if root
        else Path(CONFIG["layer2"]["ledger"])
    )
    meter = executors.Meter(
        ledger if ledger.is_absolute() else REPO_ROOT / ledger, guard=args.guard
    )
    if root:
        update_status(
            root,
            state="running",
            phase="layer2",
            layer2="running",
            figures="pending" if args.figures else "deferred",
        )
    try:
        from tqdm import tqdm

        conditions = tqdm(conditions, desc="layer 2", unit="condition", dynamic_ncols=True)
    except ImportError:
        pass
    try:
        for condition in conditions:
            judge = executors.make_judge(
                model,
                meter,
                audit_path=condition / ".branchpoint/provider_calls.jsonl",
                cond=f"std_{condition.name}",
                condition=condition.name,
            )
            extract(condition, judge, args.workers)
            standardize(condition)
            scope(condition, judge, args.workers)
            result = finalize(condition)
            print(
                f"{condition.name}: {result['n']}/{result['n_judged']} in scope "
                f"(N={result['out_of_scope']}, U={result['unclear']}) "
                f"q05-q95={result['width_q05_q95']}"
            )
    except Exception:
        if root:
            update_status(root, state="failed", phase="layer2", layer2="failed")
        raise
    if root:
        update_status(
            root,
            state="running" if args.figures else "complete",
            phase="figures" if args.figures else "layer2",
            layer2="complete",
            figures="running" if args.figures else "deferred",
        )
    if args.figures:
        try:
            subprocess.run(
                [sys.executable, "-m", "experiments.figs", "--root", str(root)],
                cwd=REPO_ROOT,
                check=True,
            )
        except Exception:
            update_status(root, state="failed", phase="figures", figures="failed")
            raise
        update_status(root, state="complete", phase="complete", figures="complete")


if __name__ == "__main__":
    main()
