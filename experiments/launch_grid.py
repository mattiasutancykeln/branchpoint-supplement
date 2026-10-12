"""Configurable, resumable experiment-matrix launcher with per-condition progress."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments import CONFIG, executors
from experiments.progress import condition_snapshot, write_status
from experiments.run_condition import PROTOCOL_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ALIASES = {values["short"]: model for model, values in CONFIG["models"].items()}
GRID = CONFIG["grid"]
DEFAULT_DATASETS = tuple(GRID["datasets"])
DEFAULT_MODELS = tuple(GRID["models"])
DEFAULT_ARMS = tuple(GRID["arms"])


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _positive(value: str, cast):
    number = cast(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _positive_int(value: str) -> int:
    return _positive(value, int)


def _positive_float(value: str) -> float:
    return _positive(value, float)


def _condition(name, dataset, requested_model, arm, n, seed, output_root) -> dict:
    if Path(name).name != name:
        raise ValueError(f"condition name must not contain a path: {name!r}")
    if arm not in ("naive", "uniform", "spread", "continuation"):
        raise ValueError(f"unknown arm {arm!r}")
    if n < 1:
        raise ValueError("condition target must be positive")
    model = MODEL_ALIASES.get(requested_model, requested_model)
    model_config = CONFIG["models"][model]
    route = model_config.get("providers", [])
    if arm in {"continuation", "spread"} and (
        provider := model_config.get("continuation_provider")
    ):
        route = [provider]
    return {
        "name": name,
        "dataset": dataset,
        "model": model,
        "arm": arm,
        "mode": "naive" if arm == "naive" else "branchpoint",
        "allocation": arm if arm in {"uniform", "spread"} else "uniform",
        "branching_method": (
            "not_applicable"
            if arm == "naive"
            else "continuation"
            if arm in {"continuation", "spread"}
            else "replay"
        ),
        "agent_interface": (
            "native" if arm in {"naive", "continuation", "spread"} else "replay"
        ),
        "judge_model": GRID["judge_model"],
        "effort": GRID["effort"],
        "workers": GRID["workers"],
        "provider_route": route,
        "seed": seed,
        "n": n,
        "out": f"{output_root}/{name}",
    }


def matrix_conditions(
    datasets=DEFAULT_DATASETS,
    models=DEFAULT_MODELS,
    arms=DEFAULT_ARMS,
    n=25,
    output_root="results/paper_v1",
) -> list[dict]:
    """Expand exactly the selected Cartesian axes; never add hidden cells."""
    return [
        _condition(
            f"{dataset}_{executors.short_name(MODEL_ALIASES.get(model, model))}_{arm}",
            dataset,
            model,
            arm,
            n,
            7,
            output_root,
        )
        for dataset in datasets
        for model in models
        for arm in arms
    ]


def explicit_condition(spec: str, output_root: str) -> dict:
    """Parse NAME:DATASET:MODEL:ARM:N:SEED."""
    try:
        name, dataset, model, arm, n, seed = spec.split(":")
        n, seed = int(n), int(seed)
    except ValueError as error:
        raise ValueError("--condition must be NAME:DATASET:MODEL:ARM:N:SEED") from error
    return _condition(name, dataset, model, arm, n, seed, output_root)


def _saved(condition: dict) -> dict:
    path = REPO_ROOT / condition["out"] / "results.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _compatible(condition: dict, saved: dict) -> bool:
    expected = {**condition, "protocol_version": PROTOCOL_VERSION}
    expected.setdefault("judge_model", "gpt-5.6-sol")
    keys = (
        "protocol_version",
        "dataset",
        "model",
        "mode",
        "allocation",
        "seed",
        "judge_model",
        "branching_method",
        "agent_interface",
        "effort",
        "workers",
        "provider_route",
    )
    return all(saved.get(key) == expected.get(key) for key in keys)


def completed(condition: dict) -> int:
    saved = _saved(condition)
    if not _compatible(condition, saved):
        return 0
    return (
        int(saved.get("n_reported", 0))
        if condition["mode"] == "naive"
        else int(saved.get("n_branches", 0))
    )


def satisfied(condition: dict) -> bool:
    saved = _saved(condition)
    if not _compatible(condition, saved):
        return False
    if saved.get("errors") or saved.get("n_failed_reports", 0):
        return False
    return completed(condition) >= condition["n"]


def env_for(condition: dict) -> dict:
    env = dict(os.environ)
    effort = condition["effort"]
    env.update(
        OPENAI_REASONING_EFFORT=effort,
        ANTHROPIC_REASONING_EFFORT=effort,
        OPENROUTER_REASONING_EFFORT=effort,
    )
    model = condition["model"]
    family = executors.providers.family(model)
    if family == "openrouter":
        env["OPENROUTER_PROVIDERS"] = ",".join(condition["provider_route"])
    return env


def command(condition: dict, args) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.run_condition",
        "--dataset",
        condition["dataset"],
        "--model",
        condition["model"],
        "--mode",
        condition["mode"],
        "--n",
        str(condition["n"]),
        "--workers",
        str(condition["workers"]),
        "--effort",
        condition["effort"],
        "--guard",
        str(args.guard),
        "--judge-model",
        args.judge_model,
        "--out",
        condition["out"],
        "--ledger",
        args.ledger,
        "--seed",
        str(condition["seed"]),
        "--allocation",
        condition["allocation"],
        "--branching-method",
        "replay" if condition["branching_method"] == "not_applicable" else condition["branching_method"],
    ]


def run_one(condition: dict, args, update) -> dict:
    output = REPO_ROOT / condition["out"]
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    started = time.time()
    with log.open("a") as stream:
        stream.write(f"\n--- launch {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        stream.flush()
        process = subprocess.Popen(
            command(condition, args),
            cwd=REPO_ROOT,
            env=env_for(condition),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            update(condition, "running", started, log)
            time.sleep(args.refresh)
        update(condition, "complete" if process.returncode == 0 else "failed", started, log)
    return {"name": condition["name"], "rc": process.returncode, "log": str(log)}


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # smallest OpenAI smoke: one naive run + one root/one Branchpoint probe
  ./runall.sh --datasets bt --models sol --arms naive,continuation --n 1 \\
    --output-root results/smoke

  # extend that exact run later; only the missing work runs
  ./runall.sh --datasets bt --models sol --arms naive,continuation --n 2 \\
    --output-root results/smoke

  # arbitrary explicit cells; repeat --condition as needed
  ./runall.sh --output-root results/paper --condition \\
    bt_sol_spread_s17:bt:sol:spread:25:17

`--n` is a target total, not an increment. Naive counts independent replicates;
Branchpoint counts probes and also has one separate root analysis. Reuse a name and
output root only to extend the same dataset/model/arm/seed.""",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--n", type=_positive_int, default=GRID["n"], help="target total per cell")
    parser.add_argument(
        "--condition",
        action="append",
        metavar="NAME:DATASET:MODEL:ARM:N:SEED",
        help="repeatable; when present, replaces --datasets/--models/--arms/--n",
    )
    parser.add_argument("--output-root", default=GRID["output_root"])
    parser.add_argument("--judge-model", default=GRID["judge_model"])
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default=GRID["effort"])
    parser.add_argument("--concurrency", type=_positive_int, default=GRID["concurrency"])
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=GRID["workers"],
        help="concurrent trajectories per Branchpoint cell and Layer-2 judge calls",
    )
    parser.add_argument("--guard", type=float, default=GRID["guard"])
    parser.add_argument("--only", help="condition-name substring")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="stop after saved Layer-1 traces and results.json; run ./layer2.sh later",
    )
    parser.add_argument("--postprocess", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--refresh", type=_positive_float, default=1.0)
    return parser


def selected_conditions(args) -> list[dict]:
    conditions = (
        [explicit_condition(spec, args.output_root) for spec in args.condition]
        if args.condition
        else matrix_conditions(
            _csv(args.datasets), _csv(args.models), _csv(args.arms), args.n, args.output_root
        )
    )
    for condition in conditions:
        model_config = CONFIG["models"][condition["model"]]
        route = model_config.get("providers", [])
        if condition["arm"] == "continuation" and (provider := model_config.get("continuation_provider")):
            route = [provider]
        condition.update(
            judge_model=args.judge_model,
            effort=args.effort,
            workers=args.workers,
            provider_route=route,
        )
    conditions = [c for c in conditions if not args.only or args.only in c["name"]]
    if not conditions:
        raise ValueError("selection contains no conditions")
    names = [condition["name"] for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("condition names must be unique")
    for condition in conditions:
        directory = REPO_ROOT / condition["out"]
        for name in ("condition.json", "results.json"):
            path = directory / name
            if path.exists():
                try:
                    saved = json.loads(path.read_text())
                except json.JSONDecodeError:
                    saved = {}
                if not _compatible(condition, saved):
                    raise ValueError(
                        f"{condition['name']} conflicts with its saved "
                        "dataset/model/arm/seed/judge"
                    )
    return conditions


def _postprocess(conditions, args) -> None:
    directories = [condition["out"] for condition in conditions]
    commands = {
        "layer2": [
            sys.executable,
            "-m",
            "experiments.layer2",
            *directories,
            "--model",
            args.judge_model,
            "--ledger",
            args.ledger,
            "--workers",
            str(args.workers),
        ],
        "figures": [sys.executable, "-m", "experiments.figs", "--root", args.output_root],
    }
    for phase, command in commands.items():
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except Exception as error:
            error.phase = phase
            raise


def _invalidate_derived(conditions, states, output_root) -> None:
    changed = [
        condition
        for condition in conditions
        if states.get(condition["name"], {}).get("state") != "satisfied"
    ]
    for condition in changed:
        directory = REPO_ROOT / condition["out"]
        for name in ("results_standardized.json", "results_final.json"):
            directory.joinpath(name).unlink(missing_ok=True)
    if changed:
        figures = REPO_ROOT / output_root / "figures"
        if figures.exists():
            shutil.rmtree(figures)


def _finalize(conditions, args, states, status_path) -> None:
    incomplete = [condition["name"] for condition in conditions if not satisfied(condition)]
    state = "incomplete" if incomplete else "complete"
    deferred = getattr(args, "generate_only", False)
    if deferred and not incomplete:
        _invalidate_derived(conditions, states, args.output_root)
    write_status(
        status_path,
        {
            "state": state,
            "phase": "layer1",
            "layer2": "deferred" if deferred else "pending",
            "figures": "deferred" if deferred else "pending",
            "updated": time.time(),
            "conditions": states,
            "incomplete": incomplete,
        },
    )
    if incomplete:
        raise SystemExit("matrix incomplete: " + ", ".join(incomplete))
    if args.postprocess and not deferred:
        try:
            _postprocess(conditions, args)
        except Exception as error:
            failed = getattr(error, "phase", "postprocessing")
            write_status(
                status_path,
                {
                    "state": "failed",
                    "phase": failed,
                    "layer2": "failed" if failed == "layer2" else "complete",
                    "figures": "failed" if failed == "figures" else "pending",
                    "updated": time.time(),
                    "conditions": states,
                },
            )
            raise
        write_status(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "layer2": "complete",
                "figures": "complete",
                "updated": time.time(),
                "conditions": states,
                "incomplete": [],
            },
        )


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        conditions = selected_conditions(args)
    except ValueError as error:
        argument_parser.error(str(error))
    args.ledger = f"{args.output_root}/costs.jsonl"
    todo = [condition for condition in conditions if not satisfied(condition)]
    todo.sort(key=lambda condition: condition["mode"] != "branchpoint")
    print(
        f"matrix: {len(conditions)} conditions; {len(conditions) - len(todo)} satisfied; "
        f"{len(todo)} to run"
    )
    for condition in todo:
        have = completed(condition)
        work = (
            f"extend {have} -> {condition['n']} ({condition['n'] - have} additional)"
            if have
            else f"fresh 0 -> {condition['n']}"
        )
        unit = "replicates" if condition["mode"] == "naive" else "probes + one root"
        print(f"  {condition['name']}: {work} {unit}")
    if args.dry_run:
        return

    status_path = REPO_ROOT / args.output_root / "run_status.json"
    ledger = REPO_ROOT / args.ledger
    states = {
        condition["name"]: {"state": "satisfied" if satisfied(condition) else "queued"}
        for condition in conditions
    }
    if not todo:
        _finalize(conditions, args, states, status_path)
        return

    lock = threading.Lock()
    bars = {}
    if not args.no_progress:
        from tqdm import tqdm

        for position, condition in enumerate(todo):
            initial = condition_snapshot(
                REPO_ROOT / condition["out"], condition["mode"], condition["n"], ledger
            )["completed"]
            bars[condition["name"]] = tqdm(
                total=condition["n"] + (condition["mode"] == "branchpoint"),
                initial=initial,
                desc=condition["name"],
                unit="analysis",
                position=position,
                leave=True,
                dynamic_ncols=True,
            )

    def update(condition, state, started, log):
        snapshot = condition_snapshot(
            REPO_ROOT / condition["out"], condition["mode"], condition["n"], ledger
        )
        row = {
            "state": state,
            "elapsed_s": round(time.time() - started, 1),
            "log": str(log),
            **snapshot,
        }
        with lock:
            states[condition["name"]] = row
            if bar := bars.get(condition["name"]):
                bar.update(max(0, snapshot["completed"] - bar.n))
                bar.set_postfix_str(
                    f"{snapshot['stage']} · calls {snapshot['calls']} · ${snapshot['usd']:.2f}"
                )
            write_status(
                status_path, {"state": "running", "updated": time.time(), "conditions": states}
            )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = [
            future.result()
            for future in as_completed([pool.submit(run_one, c, args, update) for c in todo])
        ]
    for bar in bars.values():
        bar.close()
    failed = [result for result in results if result["rc"]]
    if failed:
        write_status(
            status_path,
            {"state": "failed", "updated": time.time(), "conditions": states},
        )
        for result in failed:
            print(f"FAILED {result['name']} -> {result['log']}")
        raise SystemExit(1)
    _finalize(conditions, args, states, status_path)


if __name__ == "__main__":
    main()
