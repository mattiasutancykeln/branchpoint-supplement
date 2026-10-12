"""A files-as-state data-analysis loop.

The experiment supplies one executor. This module owns the durable transcript,
validates its decisions, runs code, and records the terminal report.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import branchpoint as bp

SUBPROCESS_TIMEOUT_S = 120
TRUNCATE_BYTES = 4096
_CONTROL_NAMES = {"snapshots", "forks", ".branchpoint", ".git", "__pycache__"}

_CONFIG = tomllib.loads(Path(__file__).with_name("protocol.toml").read_text())
REPLAY = _CONFIG["replay"]

Executor = Callable[..., str]
_DECISION_SCHEMA = json.loads(Path(__file__).with_name("decision.schema.json").read_text())


def layer1_system(mode: str) -> str:
    """Compose only the prompts visible to a dataset-analysis model."""
    # This is the Layer-1 prompt boundary. Never add Branchpoint judge or
    # Layer-2 extraction, standardization, or scope policy here.
    if mode not in {"replay", "continuation"}:
        raise ValueError(f"unknown Layer-1 mode: {mode}")
    base = _CONFIG["shared"]["system"] if mode == "replay" else _CONFIG[mode]["system"]
    parts = [base, _CONFIG["reporting"]["contract"]]
    if mode == "replay":
        parts.append(REPLAY["contract"])
    return "\n\n".join(parts)


def _fail(project: Path, note: str) -> dict:
    report = {"estimate": None, "se": None, "note": note}
    bp.Project(project).add("report", json.dumps(report))
    return report


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _validate_decision(decision: Any) -> str | None:
    if not isinstance(decision, dict):
        return "decision is not a JSON object"
    action = decision.get("action")
    if action == "report":
        if decision.get("code") is not None:
            return "report action cannot carry a code payload"
        report = decision.get("report")
        if not isinstance(report, dict):
            return "report action requires a report payload"
        if _number(report.get("estimate")) is None or _number(report.get("se")) is None:
            return "report requires numeric estimate and se"
        if not str(report.get("predictor") or "").strip():
            return "report requires predictor"
        if report.get("orientation") not in {"as_declared", "reversed", "not_applicable"}:
            return "report requires a valid orientation"
        return None
    if action != "code":
        return f"unknown action {action!r}"
    if decision.get("report") is not None:
        return "code action cannot carry a report payload"
    code = decision.get("code")
    if not isinstance(code, dict):
        return "code action requires a code payload"
    filename, script = code.get("filename"), code.get("script")
    if not isinstance(filename, str) or not filename.strip():
        return "code action requires filename"
    path = Path(filename)
    if (
        path.is_absolute()
        or path.parts[:1] != ("scripts",)
        or ".." in path.parts
        or path.suffix != ".py"
    ):
        return "filename must be a relative scripts/*.py path"
    if not isinstance(script, str) or not script.strip():
        return "code action requires script"
    return None


def _extract_json(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _context_blocks(question: str, events: list[dict]) -> list[str]:
    blocks = [f"# Research question and estimand\n\n{question.strip()}\n\n# Steps taken so far\n"]
    blocks.extend(f"\n[event i={e['i']} kind={e['kind']}] {e['content']}" for e in events)
    return blocks


def _decision(executor: Executor, system: str, blocks: list[str], turn: str):
    attempts = REPLAY["retries"]
    prompt = turn
    failures = []
    for _ in range(attempts):
        raw = (
            executor(system, blocks, prompt)
            if getattr(executor, "wants_context_blocks", False)
            else executor(system, "".join(blocks) + prompt)
        )
        failures.append(raw)
        value = _extract_json(raw)
        error = _validate_decision(value)
        if error is None:
            return value
        prompt = f"{turn}\nYour previous response was invalid: {error}. Return only JSON."
    return failures


def _files(project: Path) -> dict[str, int]:
    files = {}
    for path in project.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(project)
        if relative.name == "trace.jsonl":
            continue
        if relative.parts and relative.parts[0] in _CONTROL_NAMES:
            continue
        files[str(relative)] = path.stat().st_mtime_ns
    return files


def _truncate(text: str) -> str:
    raw = text.encode(errors="replace")
    if len(raw) <= TRUNCATE_BYTES:
        return text
    return raw[:TRUNCATE_BYTES].decode(errors="ignore") + "...[truncated]"


def _record_code(project: Path, result: dict) -> None:
    content = json.dumps(
        {
            **result,
            "stdout": _truncate(result["stdout"]),
            "stderr": _truncate(result["stderr"]),
        }
    )
    bp.Project(project).add("code", content, [result["filename"]], result["outputs"])


def _run_code(
    project: Path,
    filename: str,
    script: str,
    before_record: Callable[[dict], None] = lambda _result: None,
) -> dict:
    target = project / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    before = _files(project)
    target.write_text(script)
    try:
        result = subprocess.run(
            [sys.executable, filename],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        code, stdout, stderr, timed_out = result.returncode, result.stdout, result.stderr, False
    except subprocess.TimeoutExpired as error:
        code = None
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stderr += f"\n[TIMEOUT after {SUBPROCESS_TIMEOUT_S}s]"
        timed_out = True
    outputs = sorted(path for path, stamp in _files(project).items() if before.get(path) != stamp)
    result = {
        "filename": filename,
        "returncode": code,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "outputs": outputs,
    }
    before_record(result)
    _record_code(project, result)
    return result


def run_loop(project_dir: str | Path, executor: Executor, max_steps: int | None = None) -> dict:
    project = Path(project_dir)
    question_path = project / "QUESTION.md"
    if not question_path.exists():
        raise FileNotFoundError(f"{question_path} not found")
    limit = max_steps if max_steps is not None else REPLAY["steps"]
    system = layer1_system("replay")

    for _ in range(limit):
        events = bp.Project(project).events()
        blocks = _context_blocks(question_path.read_text(), events)
        empty = "\n(none yet -- this is the first decision)" if not events else ""
        turn = f"{empty}\n\n# Your turn\n{REPLAY['turn']}"
        decision = _decision(executor, system, blocks, turn)
        if not isinstance(decision, dict):
            with (project / "executor_debug.log").open("a") as debug:
                for failure in decision:
                    debug.write(f"--- failed attempt ---\n{failure}\n")
            return _fail(project, "executor returned no valid decision; see executor_debug.log")
        bp.Project(project).add("llm", json.dumps(decision))
        if decision["action"] == "code":
            try:
                _run_code(project, **decision["code"])
            except Exception as error:
                return _fail(project, f"code step crashed: {type(error).__name__}: {error}")
            continue
        payload = decision["report"]
        note = payload.get("note") or ""
        estimate, se = _number(payload.get("estimate")), _number(payload.get("se"))
        report = {
            "estimate": estimate,
            "se": se,
            "note": note,
            "predictor": payload.get("predictor"),
            "orientation": payload.get("orientation"),
        }
        bp.Project(project).add("report", json.dumps(report))
        return report
    return _fail(project, f"max_steps ({limit}) exceeded without a report")


def run_agent(project_dir: str | Path, executor_fn: Executor, max_steps: int | None = None) -> dict:
    """Run or resume an analysis; unexpected host failures become durable reports."""
    project = Path(project_dir)
    try:
        return run_loop(project, executor_fn, max_steps)
    except Exception as error:
        return _fail(project, f"run_agent crashed: {type(error).__name__}: {error}")
