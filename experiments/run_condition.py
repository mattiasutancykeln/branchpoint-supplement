"""One runner for every cell of the grid.

    python -m experiments.run_condition --dataset bt --model claude-sonnet-5 \
        --mode branchpoint --n 50 [--workers 8] [--guard 350]

modes: naive (N independent unassisted runs), branchpoint (one root, then
forced branches to n).

Writes <out>/results.json -- the numerics that ARE the result: one row per
analysis with estimate, se, and note, plus the condition's metadata and
its metered cost. Traces stay on disk as working state, untracked."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import branchpoint as bp
from experiments import CONFIG, datasets, executors, layer2, write_json
from harness import agent, continuation, hygiene

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = 6


def materialize(dst: Path, ds: dict) -> Path:
    """Repair every missing component of a condition's starting state."""
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "data.csv").exists():
        shutil.copy(REPO_ROOT / ds["data"], dst / "data.csv")
    if not (dst / "QUESTION.md").exists():
        q = (REPO_ROOT / ds["question"]).read_text()
        if hits := hygiene.screen(q):
            raise ValueError(f"{ds['question']} fails the hygiene screen: {hits}")
        (dst / "QUESTION.md").write_text(q)
    (dst / "trace.jsonl").touch()
    return dst


def run_seeds(out: Path, ds, model, meter, n, workers) -> None:
    """Run N independent native trajectories, without intervention."""

    def one(k: int):
        d = materialize(out / f"seed{k:03d}", ds)
        if bp.Project(d).report() is not None:
            return  # finished seeds are never re-run
        continuation.run_agent(
            d,
            model,
            before_call=meter.check,
            observe_usage=lambda call_id, phase, usage: meter.book(
                model,
                phase,
                usage,
                cond=out.name,
                seed=k,
                role_tag="exec",
                call_id=call_id,
            ),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(n)))


def run_branchpoint(
    out: Path,
    ds,
    model,
    meter,
    n,
    workers,
    judge_model,
    seed: int = 7,
    allocation="uniform",
    branching_method="replay",
) -> dict:
    """One root run, then forced branches until n branches are probed."""
    proj = materialize(out, ds)
    if branching_method == "continuation":

        def resume(path):
            return continuation.run_loop(
                path,
                model,
                before_call=meter.check,
                observe_usage=lambda call_id, phase, usage: meter.book(
                    model, phase, usage, cond=out.name, role_tag="exec", call_id=call_id
                ),
            )

    else:
        ex = executors.make_executor(
            model,
            meter,
            audit_path=out / ".branchpoint/provider_calls.jsonl",
            cond=out.name,
            role_tag="exec",
        )

        def resume(path):
            return agent.run_loop(path, ex)
    if bp.Project(proj).report() is None:
        if branching_method == "continuation":
            continuation.run_agent(
                proj,
                model,
                before_call=meter.check,
                observe_usage=lambda call_id, phase, usage: meter.book(
                    model, phase, usage, cond=out.name, role_tag="exec", call_id=call_id
                ),
            )
        else:
            agent.run_agent(proj, executor_fn=ex)
    root_report = bp.Project(proj).report()
    if not root_report or root_report.get("estimate") is None:
        raise RuntimeError(f"root analysis failed: {(root_report or {}).get('note', 'no report')}")
    judge = executors.make_judge(
        judge_model,
        meter,
        audit_path=out / ".branchpoint/provider_calls.jsonl",
        cond=out.name,
        condition=out.name,
        segment="sampler",
    )
    # Layer 2 standardizes and scopes both arms after the grid. Sampling ends
    # here, with raw reports and branches on disk.
    summary = bp.sample(
        proj,
        resume,
        judge,
        stop_fn=meter.check,
        workers=workers,
        cap=n,
        strikes=2,
        seed=seed,
        allocation=allocation,
        branching_method=branching_method,
    )
    if summary["errors"]:
        raise RuntimeError(f"{len(summary['errors'])} probe(s) failed; inspect bp_state.json")
    return summary


def collect(out: Path, mode: str, ds, model, n, meter, extra=None) -> dict:
    """The numerics that ARE the result."""
    if mode == "branchpoint":
        # carry the declaration through: dropping it here is what made the
        # structural orientation field look empty downstream
        keep = ("estimate", "se", "note", "predictor", "orientation")
        rows = [
            {"branch": b["branch"], **{k: b.get(k) for k in keep}} for b in layer2.branches(out)
        ]
    else:
        rows = []
        for k in range(n):
            d = out / f"seed{k:03d}"
            rep = bp.Project(d).report() if (d / "trace.jsonl").exists() else None
            rows.append(
                {
                    "seed": k,
                    **{
                        key: (rep or {}).get(key)
                        for key in ("estimate", "se", "note", "predictor", "orientation")
                    },
                }
            )
    ok = [r for r in rows if r.get("estimate") is not None]
    ests = sorted(r["estimate"] for r in ok)
    uniq = len({round(e, 3) for e in ests})

    spent = 0.0
    if meter.path.exists():
        spent = sum(
            row.get("usd", 0.0)
            for line in meter.path.read_text().splitlines()
            if line.strip()
            and (row := json.loads(line)).get("cond") == out.name
        )
    res = {
        **(extra or {}),
        "mode": mode,
        "dataset": ds["id"],
        "model": model,
        "n_requested": n,
        "n_reported": len(ok),
        "rows": rows,
        "envelope": [ests[0], ests[-1]] if ests else None,
        "width": (ests[-1] - ests[0]) if len(ests) > 1 else None,
        # min/max is dominated by a single extreme branch; the quantile
        # spread is the stable summary of the same draws.
        "q05_q95": (
            [layer2.quantile(ests, 0.05), layer2.quantile(ests, 0.95)]
            if len(ests) > 1
            else None
        ),
        "width_q05_q95": (
            layer2.quantile(ests, 0.95) - layer2.quantile(ests, 0.05)
            if len(ests) > 1
            else None
        ),
        "iqr": (
            [layer2.quantile(ests, 0.25), layer2.quantile(ests, 0.75)]
            if len(ests) > 1
            else None
        ),
        "width_iqr": (
            layer2.quantile(ests, 0.75) - layer2.quantile(ests, 0.25)
            if len(ests) > 1
            else None
        ),
        "median": layer2.quantile(ests, 0.5) if ests else None,
        "unique_ends": uniq,
        "mean": sum(ests) / len(ests) if ests else None,
        "usd": spent,
        "n_failed_reports": len(rows) - len(ok),
    }
    write_json(out / "results.json", res)
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "mode",
                    "dataset",
                    "model",
                    "n_reported",
                    "envelope",
                    "width",
                    "width_q05_q95",
                    "width_iqr",
                    "median",
                    "unique_ends",
                    "usd",
                )
            },
            indent=2,
        )
    )
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", required=True, choices=["naive", "branchpoint"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--judge-model", default="claude-sonnet-5")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="low")
    p.add_argument("--guard", type=float, default=500.0)
    p.add_argument("--allocation", choices=["uniform", "spread"], default="uniform")
    p.add_argument(
        "--branching-method",
        choices=["replay", "continuation"],
        default="replay",
        help="replay rebuilds visible context; continuation preserves provider-native history",
    )
    p.add_argument(
        "--seed", type=int, default=7, help="branchpoint draw seed; replicates vary only this"
    )
    p.add_argument("--out", default=None)
    p.add_argument("--ledger", help="shared JSONL cost ledger (default: beside --out)")
    a = p.parse_args()

    os.environ.update(
        OPENAI_REASONING_EFFORT=a.effort,
        ANTHROPIC_REASONING_EFFORT=a.effort,
        OPENROUTER_REASONING_EFFORT=a.effort,
    )

    ds = datasets.get(a.dataset)
    short = executors.short_name(a.model)
    suffix = f"_{a.branching_method}" if a.mode == "branchpoint" else ""
    out = REPO_ROOT / (a.out or f"results/grid/{a.dataset}_{short}_{a.mode}{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.json"
    old = json.loads(result_path.read_text()) if result_path.exists() else {}
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": ds["id"],
        "model": a.model,
        "mode": a.mode,
        "allocation": a.allocation,
        "seed": a.seed,
        "judge_model": a.judge_model,
        "effort": a.effort,
        "workers": a.workers,
        "provider_route": CONFIG["models"][a.model].get("providers", []),
        "branching_method": a.branching_method if a.mode == "branchpoint" else "not_applicable",
        "agent_interface": (
            "native" if a.mode == "naive" or a.branching_method == "continuation" else "replay"
        ),
    }
    manifest = out / "condition.json"
    prior = json.loads(manifest.read_text()) if manifest.exists() else old
    if prior:
        wrong = [key for key, value in identity.items() if prior.get(key) != value]
        if wrong:
            detail = ", ".join(
                f"{key}={prior.get(key)!r} (requested {identity[key]!r})" for key in wrong
            )
            raise SystemExit(f"cannot extend incompatible condition at {out}: {detail}")
        if a.n < old.get("n_requested", 0):
            raise SystemExit(f"cannot shrink {out}; choose a new output directory")
    write_json(manifest, identity)
    ledger = Path(a.ledger) if a.ledger else out.parent / "costs.jsonl"
    if not ledger.is_absolute():
        ledger = REPO_ROOT / ledger
    meter = executors.Meter(ledger, guard=a.guard)
    t0 = time.time()

    extra = dict(identity)
    if a.mode == "branchpoint":
        extra.update(
            run_branchpoint(
                out,
                ds,
                a.model,
                meter,
                a.n,
                a.workers,
                a.judge_model,
                seed=a.seed,
                allocation=a.allocation,
                branching_method=a.branching_method,
            )
        )
    else:
        run_seeds(out, ds, a.model, meter, a.n, a.workers)
    elapsed = round(time.time() - t0, 1)
    previous = old.get("n_requested", 0)
    extensions = list(old.get("extensions", []))
    if not old or a.n > previous:
        extensions.append({"from": previous, "to": a.n, "wall_clock_s": elapsed})
    extra["extensions"] = extensions
    extra["wall_clock_s"] = round(old.get("wall_clock_s", 0.0) + elapsed, 1)
    collect(out, a.mode, ds, a.model, a.n, meter, extra)


if __name__ == "__main__":
    main()
