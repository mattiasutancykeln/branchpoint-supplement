"""Auditable native continuation with analysis and action in separate turns."""

from __future__ import annotations

import hashlib
import json
import os
import time
import tomllib
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import branchpoint as bp
from harness import agent, providers

_PROTOCOL = tomllib.loads(Path(__file__).with_name("protocol.toml").read_text())
_CONFIG = _PROTOCOL["continuation"]
# Native continuation changes transport/history, not the Layer-1 prompt boundary.
_SYSTEM = agent.layer1_system("continuation")
_TOOLS = json.loads(Path(__file__).with_name("action.tools.json").read_text())


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Audit:
    """Append requests before I/O and retain every response or failure unchanged."""

    def __init__(self, project: Path):
        self.path = project / bp.CALLS
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, row: dict) -> None:
        with self.path.open("a") as file:
            file.write(json.dumps(row) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def start(self, event_i: int, phase: str, model: str, request: dict) -> str:
        call_id = uuid.uuid4().hex
        self._append(
            {
                "kind": "request",
                "call_id": call_id,
                "event_i": event_i,
                "phase": phase,
                "model": model,
                "created_at_ns": time.time_ns(),
                "sha256": _digest(request),
                "request": request,
            }
        )
        return call_id

    def finish(
        self,
        call_id: str,
        event_i: int,
        status: str,
        response: dict | None = None,
        error: str | None = None,
    ) -> None:
        self._append(
            {
                "kind": "response",
                "call_id": call_id,
                "event_i": event_i,
                "status": status,
                "created_at_ns": time.time_ns(),
                "sha256": _digest(response) if response is not None else None,
                "response": response,
                "error": error,
            }
        )

    def commit(self, call_id: str, event_i: int) -> None:
        self._append(
            {
                "kind": "commit",
                "call_id": call_id,
                "event_i": event_i,
                "created_at_ns": time.time_ns(),
            }
        )

    def tool_start(self, event_i: int, call: dict) -> None:
        self._append(
            {
                "kind": "tool_start",
                "call_id": call["id"],
                "event_i": event_i,
                "created_at_ns": time.time_ns(),
                "arguments": call["arguments"],
            }
        )

    def tool_result(self, event_i: int, call_id: str, result: dict) -> None:
        self._append(
            {
                "kind": "tool_result",
                "call_id": call_id,
                "event_i": event_i,
                "created_at_ns": time.time_ns(),
                "sha256": _digest(result),
                "result": result,
            }
        )

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return [json.loads(line) for line in self.path.read_text().splitlines() if line]
        except json.JSONDecodeError as error:
            raise RuntimeError(f"corrupt continuation audit: {self.path}") from error

    def orphan_request(self) -> dict | None:
        rows = self.rows()
        answered = {row["call_id"] for row in rows if row["kind"] == "response"}
        return next(
            (
                row
                for row in reversed(rows)
                if row["kind"] == "request" and row["call_id"] not in answered
            ),
            None,
        )

    def last_exchange(self) -> tuple[dict, dict] | None:
        rows = self.rows()
        requests = {row["call_id"]: row for row in rows if row["kind"] == "request"}
        responses = {row["call_id"]: row for row in rows if row["kind"] == "response"}
        commits = [row for row in rows if row["kind"] == "commit"]
        if not commits:
            return None
        response = responses[commits[-1]["call_id"]]
        request = requests[commits[-1]["call_id"]]
        return request["request"], response["response"]

    def pending_response(self) -> tuple[dict, dict] | None:
        rows = self.rows()
        committed = {row["call_id"] for row in rows if row["kind"] == "commit"}
        requests = {row["call_id"]: row for row in rows if row["kind"] == "request"}
        pending = [
            row
            for row in rows
            if row["kind"] == "response"
            and row["status"] == "received"
            and row["call_id"] not in committed
        ]
        if not pending:
            return None
        response = pending[-1]
        return requests[response["call_id"]], response

    def result_for(self, event_i: int) -> dict:
        return next(
            row["result"]
            for row in reversed(self.rows())
            if row["kind"] == "tool_result" and row["event_i"] == event_i
        )

    def tool_state(self, call_id: str) -> tuple[str, dict | None]:
        rows = [row for row in self.rows() if row.get("call_id") == call_id]
        results = [row["result"] for row in rows if row["kind"] == "tool_result"]
        if results:
            return "complete", results[-1]
        return ("uncertain", None) if any(row["kind"] == "tool_start" for row in rows) else ("new", None)


def _exchange(
    project: Path,
    model: str,
    phase: str,
    client: dict | list[dict],
    tools: list[dict] | None,
    before_call: Callable[[], None],
    observe_usage: Callable[[str, str, dict], None],
) -> tuple[str, str, list[dict]]:
    audit = Audit(project)
    previous = audit.last_exchange()
    history = providers.native_history(model, *previous) if previous else []
    request = providers.native_request(
        model, _SYSTEM, history, client, tools, require_tool=phase == "action"
    )
    event_i = len(bp.Project(project).events())
    attempts = 5
    for attempt in range(attempts):
        before_call()
        call_id = audit.start(event_i, phase, model, request)
        response = None
        usage = None
        try:
            response = providers.native_complete(model, request)
            usage = providers.native_usage(model, response)
            text, calls, usage = providers.native_result(model, response)
            if phase == "analysis" and (calls or not text.strip()):
                raise ValueError("analysis turn must return visible text and no tool call")
            if phase == "action" and len(calls) != 1:
                raise ValueError("action turn must return exactly one tool call")
        except Exception as error:
            audit.finish(
                call_id,
                event_i,
                "rejected",
                response,
                f"{type(error).__name__}: {error}",
            )
            if usage is not None:
                observe_usage(call_id, phase, usage)
            if isinstance(error, providers.ProviderError) and attempt < attempts - 1:
                time.sleep(2**attempt)
                continue
            raise
        break
    audit.finish(call_id, event_i, "received", response)
    observe_usage(call_id, phase, usage)
    _record_response(project, event_i, phase, call_id, text, calls)
    return call_id, text, calls


def _record_response(
    project: Path,
    event_i: int,
    phase: str,
    call_id: str,
    text: str,
    calls: list[dict],
) -> None:
    trace = bp.Project(project)
    existing = [event for event in trace.events() if event.get("call_id") == call_id]
    if not existing:
        content = text if phase == "analysis" else json.dumps(calls[0])
        trace.add(phase, content, call_id=call_id, boundary="after_response")
    elif existing[0]["i"] != event_i:
        raise ValueError(f"call {call_id} is linked to the wrong trace event")
    Audit(project).commit(call_id, event_i)


def _recover_response(
    project: Path,
    model: str,
    observe_usage: Callable[[str, str, dict], None],
) -> None:
    audit = Audit(project)
    if orphan := audit.orphan_request():
        raise RuntimeError(
            f"provider call {orphan['call_id']} has an unknown outcome; refusing to duplicate it"
        )
    pending = audit.pending_response()
    if not pending:
        return
    request, received = pending
    event_i = request["event_i"]
    if event_i != len(bp.Project(project).events()):
        linked = [
            event
            for event in bp.Project(project).events()
            if event.get("call_id") == request["call_id"]
        ]
        if not linked:
            raise ValueError("received provider response conflicts with durable trace")
    text, calls, usage = providers.native_result(model, received["response"])
    observe_usage(request["call_id"], request["phase"], usage)
    _record_response(project, event_i, request["phase"], request["call_id"], text, calls)


def _action(call: dict) -> dict:
    if call["name"] == "run_code":
        decision = {"action": "code", "basis": None, "code": call["arguments"], "report": None}
    elif call["name"] == "submit_report":
        decision = {
            "action": "report",
            "basis": None,
            "code": None,
            "report": call["arguments"],
        }
    else:
        raise ValueError(f"unknown tool {call['name']!r}")
    if error := agent._validate_decision(decision):
        raise ValueError(error)
    return decision


def run_loop(
    project_dir: str | Path,
    model: str,
    max_steps: int | None = None,
    before_call: Callable[[], None] = lambda: None,
    observe_usage: Callable[[str, str, dict], None] = lambda _id, _phase, _usage: None,
) -> dict:
    """Resume the proposal → typed action → tool-result state machine."""
    project = Path(project_dir)
    question = project / "QUESTION.md"
    if not question.exists():
        raise FileNotFoundError(f"{question} not found")
    limit = max_steps or _CONFIG["steps"]

    while True:
        _recover_response(project, model, observe_usage)
        events = bp.Project(project).events()
        if report := bp.Project(project).report():
            return report
        last = events[-1] if events else None

        if last is None or last["kind"] == "code":
            if last is None:
                client: dict | list[dict] = {
                    "kind": "text",
                    "content": _CONFIG["first"].format(question=question.read_text().strip()),
                }
            else:
                action_event = next(
                    event for event in reversed(events[:-1]) if event["kind"] == "action"
                )
                call = json.loads(action_event["content"])
                result = Audit(project).result_for(last["i"])
                client = [
                    {
                        "kind": "tool_result",
                        "call_id": call["id"],
                        "content": json.dumps(result),
                    },
                    {"kind": "text", "content": _CONFIG["analyze"]},
                ]
            _call_id, _text, _calls = _exchange(
                project, model, "analysis", client, _TOOLS, before_call, observe_usage
            )
            continue

        if last["kind"] in {"analysis", "intervention"}:
            if len([event for event in events if event["kind"] == "action"]) >= limit:
                return agent._fail(project, f"max_steps ({limit}) exceeded without a report")
            intervention = f"{last['content']}\n\n" if last["kind"] == "intervention" else ""
            _call_id, _text, _calls = _exchange(
                project,
                model,
                "action",
                {"kind": "text", "content": intervention + _CONFIG["act"]},
                _TOOLS,
                before_call,
                observe_usage,
            )
            continue

        if last["kind"] == "action":
            call = json.loads(last["content"])
            decision = _action(call)
            if decision["action"] == "report":
                report = decision["report"]
                bp.Project(project).add("report", json.dumps(report))
                return report
            audit = Audit(project)
            state, result = audit.tool_state(call["id"])
            if state == "uncertain":
                raise RuntimeError(f"tool {call['id']} has an uncertain external effect")
            if state == "complete":
                agent._record_code(project, result)
                continue
            audit.tool_start(last["i"], call)
            code_i = last["i"] + 1

            def record(value, audit=audit, code_i=code_i, call_id=call["id"]):
                audit.tool_result(code_i, call_id, value)

            agent._run_code(
                project,
                **decision["code"],
                before_record=record,
            )
            continue

        raise ValueError(f"cannot resume continuation after {last['kind']!r}")

def run_agent(project_dir: str | Path, model: str, **kwargs) -> dict:
    """Run or resume; provider and host failures remain durable and auditable."""
    try:
        return run_loop(project_dir, model, **kwargs)
    except Exception as error:
        return agent._fail(Path(project_dir), f"continuation crashed: {type(error).__name__}: {error}")
