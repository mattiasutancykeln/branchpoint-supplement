"""Fail-fast readiness checks for exactly the selected experiment matrix."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from experiments import datasets, executors, launch_grid

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = launch_grid.parser()
    args, _unknown = parser.parse_known_args()
    try:
        conds = launch_grid.selected_conditions(args)
    except ValueError as error:
        parser.error(str(error))
    models = {c["model"] for c in conds} | {args.judge_model}
    selected_data = {c["dataset"] for c in conds}
    failures = []

    families = {executors.providers.family(model) for model in models}
    modules = {"matplotlib", "numpy", "pandas", "scipy", "statsmodels", "tqdm"}
    modules |= {"openai" for family in families if family in {"openai", "openrouter"}}
    modules |= {"anthropic" for family in families if family == "anthropic"}
    for module in sorted(modules):
        try:
            importlib.import_module(module)
        except ImportError:
            failures.append(f"missing Python dependency: {module}")

    for ds_id in sorted(selected_data):
        ds = datasets.get(ds_id)
        for label in ("data", "question"):
            rel = ds[label]
            path = REPO_ROOT / rel
            if not path.is_file() or not path.stat().st_size:
                failures.append(f"missing {ds_id} {label}: {rel}")

    requirements = {executors.providers.family(model): model for model in sorted(models)}
    env_by_family = {"openai": ("OPENAI_API_KEY",),
                     "anthropic": ("ANTHROPIC_API_KEY",),
                     "openrouter": ("OPENROUTER_API_KEY",),
                     "google": ("GEMINI_API_KEY",)}
    for fam, model in requirements.items():
        for name in env_by_family[fam]:
            if not os.environ.get(name):
                failures.append(f"{name} is required for selected model {model}")

    if failures:
        print(f"\nnot ready: {len(failures)} blocker(s)")
        for failure in failures:
            print(f"  - {failure}")
        if args.dry_run:
            print("\ndry-run: showing the plan anyway; a real run would stop here")
            return
        raise SystemExit(2)


if __name__ == "__main__":
    main()
