"""Read-only progress snapshots for running or resumed experiment cells."""
from __future__ import annotations

import json
import os
from pathlib import Path

from branchpoint import Project


def _terminal(project: Path, segment: str = "") -> bool:
    try:
        return Project(project, segment).report() is not None
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _analysis_stage(project: Path, segment: str = "") -> str:
    try:
        events = Project(project, segment).events()
    except (FileNotFoundError, json.JSONDecodeError):
        events = []
    if not events:
        return "waiting for model call 1"
    llms = [e for e in events if e["kind"] == "llm"]
    codes = [e for e in events if e["kind"] == "code"]
    if events[-1]["kind"] == "report":
        return "reported"
    if events[-1]["kind"] == "code":
        return f"awaiting model call {len(llms) + 1} after code step {len(codes)}"
    if events[-1]["kind"] == "intervention":
        return "waiting for fork model call 1"
    if events[-1]["kind"] == "llm":
        try:
            decision = json.loads(events[-1]["content"])
        except (json.JSONDecodeError, TypeError):
            decision = {}
        if decision.get("action") == "code":
            return f"running code step {len(codes) + 1}: {decision.get('filename')}"
    return f"processing trace event {events[-1]['kind']}"


def _judge_stage(rows: list[dict]) -> str:
    last = next((r for r in reversed(rows) if r.get("role") == "judge"), None)
    if not last:
        return "finding decisions"
    instrument = last.get("instrument") or "judge"
    return f"decision mapping; last judge: {instrument}"


def condition_snapshot(out: Path, mode: str, n: int, ledger: Path) -> dict:
    if mode == "naive":
        completed = sum(_terminal(out / f"seed{k:03d}") for k in range(n))
        active = sum((out / f"seed{k:03d}/trace.jsonl").exists()
                     and not _terminal(out / f"seed{k:03d}") for k in range(n))
        total, root = n, False
        active_seeds = [k for k in range(n)
                        if (out / f"seed{k:03d}/trace.jsonl").exists()
                        and not _terminal(out / f"seed{k:03d}")]
    else:
        root = _terminal(out)
        forks = list(out.glob("forks/bp/*/trace.jsonl"))
        completed = int(root) + sum(
            _terminal(out, str(p.parent.relative_to(out))) for p in forks)
        active = sum(not _terminal(out, str(p.parent.relative_to(out))) for p in forks)
        total = n + 1

    calls = in_tokens = out_tokens = cached = 0
    rows = []
    usd = 0.0
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if row.get("cond") != out.name:
                continue
            rows.append(row)
            calls += 1
            usd += row.get("usd", 0.0)
            in_tokens += row.get("in_tokens", 0)
            out_tokens += row.get("out_tokens", 0)
            cached += row.get("cached_in", 0)
    if mode == "naive":
        if active_seeds:
            k = active_seeds[0]
            stage = f"seed {k + 1}/{n}; {_analysis_stage(out / f'seed{k:03d}')}"
        else:
            stage = f"{completed}/{n} seeds reported"
    elif not root:
        stage = "root; " + _analysis_stage(out)
    else:
        state_path = out / "bp_state.json"
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
        except json.JSONDecodeError:
            state = {}  # sampler may be between truncate and rewrite
        in_flight = state.get("in_flight", {})
        attempted = state.get("n_branches", 0)
        if in_flight:
            seg = next(iter(in_flight.values()))
            detail = _analysis_stage(out, seg)
            stage = f"branch {attempted}/{n}; {detail}"
        elif attempted >= n:
            stage = f"branches {attempted}/{n}; finishing sampler"
        else:
            stage = f"root reported; {_judge_stage(rows)}"
    last_call = ({k: rows[-1].get(k) for k in ("role", "instrument", "model", "usd")}
                 if rows else None)
    return {"completed": min(completed, total), "total": total,
            "active": active, "root_complete": root, "calls": calls,
            "usd": usd, "input_tokens": in_tokens,
            "output_tokens": out_tokens, "cached_input_tokens": cached,
            "cache_rate": cached / in_tokens if in_tokens else 0.0,
            "stage": stage, "last_call": last_call}


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
