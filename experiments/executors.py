"""Provider routing, prompt caching, and metering for the experiment grid.

One place decides how a model is called and what it costs. Anthropic models
get EXPLICIT prompt caching (per-event content blocks, cache_control on the
system block and on the last context block) via the agent loop's
`wants_context_blocks` executor path; OpenAI and OpenRouter models auto-cache
server-side and report the cached share back, which is billed at the cached
rate. Every call appends one metering row with its full token breakdown.

Caching is output-invariant: providers.complete_json_cached asserts the
concatenated block text equals the plain-call prompt byte for byte, so the
model sees identical bytes either way -- only billing changes."""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import uuid
from pathlib import Path

from experiments import CONFIG
from harness import agent, providers

# USD per million tokens: (input, output, cached_input, cache_write).
# cached_input is what a cache READ costs; cache_write is the surcharged
# first write (Anthropic 1.25x input; auto-caching providers do not charge
# a write premium, so cache_write == input there).
RATES = {model: tuple(values["rates"]) for model, values in CONFIG["models"].items()}


# Directory-safe short name per model. Explicit rather than derived: the
# derivation that worked for "claude-sonnet-5" produced "5.3" for
# "openrouter/z-ai/glm-5.3".
SHORT = {model: values["short"] for model, values in CONFIG["models"].items()}


def short_name(model: str) -> str:
    if model in SHORT:
        return SHORT[model]
    base = model.split("/")[-1]
    parts = [x for x in base.split("-") if not x.replace(".", "").isdigit()]
    return "".join(parts) or base


def cost(model: str, u: dict) -> float:
    """USD for one call from its usage dict. Cached reads and cache writes
    are billed at their own rates; plain input excludes both."""
    r_in, r_out, r_cached, r_write = RATES[model]
    cached = u.get("cache_read_input_tokens", 0) or u.get("cached_input_tokens", 0)
    written = u.get("cache_creation_input_tokens", 0)
    # Anthropic reports input_tokens EXCLUDING cache reads/writes; OpenAI and
    # OpenRouter report a total that INCLUDES the cached share.
    plain = u.get("input_tokens", 0)
    if u.get("cached_input_tokens"):
        plain = max(plain - cached, 0)
    return (
        plain * r_in + cached * r_cached + written * r_write + u.get("output_tokens", 0) * r_out
    ) / 1e6


class Meter:
    """Thread-safe append-only cost ledger; also the budget guard."""

    def __init__(self, path: str | Path, guard: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.guard = guard
        self.lock = threading.Lock()
        self.spent = self._replay()

    def _replay(self) -> float:
        return sum(row["usd"] for row in self._rows())

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def check(self) -> None:
        with self.lock, self.lock_path.open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            self.spent = max(self.spent, self._replay())
            stopped = self.guard is not None and self.spent >= self.guard
        if stopped:
            raise SystemExit(f"BUDGET GUARD STOP (${self.spent:.2f} >= ${self.guard})")

    def book(self, model: str, role: str, u: dict, **tags) -> float:
        usd = cost(model, u)
        row = {
            "t": time.time(),
            "model": model,
            "role": role,
            "usd": usd,
            "in_tokens": u.get("input_tokens", 0),
            "out_tokens": u.get("output_tokens", 0),
            "cached_in": u.get("cache_read_input_tokens", 0) or u.get("cached_input_tokens", 0),
            "cache_write": u.get("cache_creation_input_tokens", 0),
            **tags,
        }
        with self.lock, self.lock_path.open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            rows = self._rows()
            self.spent = max(self.spent, sum(item["usd"] for item in rows))
            if (call_id := tags.get("call_id")) and (
                prior := next((item for item in rows if item.get("call_id") == call_id), None)
            ):
                return prior["usd"]
            self.spent += usd
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return usd


class CallAudit:
    """Process-safe, durable provider request/response ledger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def __call__(self, row: dict) -> None:
        row = {"created_at_ns": time.time_ns(), **row}
        with self.lock_path.open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            with self.path.open("a") as file:
                file.write(json.dumps(row) + "\n")
                file.flush()
                os.fsync(file.fileno())


def _log_prompt(role: str, system: str, prompt: str, **tags) -> None:
    """Env-gated verbatim capture of what was actually sent. Off by default;
    set BP_PROMPT_LOG to a path to make every prompt auditable -- the only
    way to verify that no reference result, comparison set, or steering
    ever reaches a model."""
    path = os.environ.get("BP_PROMPT_LOG")
    if not path:
        return
    with _LOG_LOCK, open(path, "a") as f:
        f.write(
            json.dumps({"t": time.time(), "role": role, "system": system, "prompt": prompt, **tags})
            + "\n"
        )


_LOG_LOCK = threading.Lock()


# Truncation is spelled differently by each provider: OpenAI reports an
# 'incomplete' status, Anthropic a stop_reason of 'max_tokens'. Missing the
# Anthropic spelling cost a completed 50-branch run its inference pass.
_TRUNCATED = ("incomplete", "max_tokens", "non-JSON")


# Network-level failures that say nothing about the request: retried with
# backoff rather than killing a run. A dropped connection cost a completed
# revalidation pass before this existed.
_TRANSIENT = (
    "ReadError",
    "ConnectError",
    "ReadTimeout",
    "WriteError",
    "RemoteProtocolError",
    "ConnectionResetError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "OverloadedError",
    "Connection reset",
    "overloaded_error",
)


def _call_plain(
    model,
    system,
    prompt,
    schema,
    max_tokens,
    attempts: int = 4,
    *,
    meter=None,
    role=None,
    audit=None,
    tags=None,
):
    """Plain call with two retry ladders: truncation escalates the output
    cap (reasoning tokens count against it on gpt-5.x, and a long chunked
    judge answer can exceed it on any provider); a transient network fault
    is retried with backoff."""
    last = None
    tags = tags or {}
    for attempt in range(attempts):
        for cap in (max_tokens, max_tokens * 3):
            call_id = uuid.uuid4().hex
            try:
                kwargs = (
                    {"audit": audit, "call_id": call_id, "tags": tags}
                    if audit is not None
                    else {}
                )
                result = providers.complete_json(
                    model, system, prompt, schema, max_tokens=cap, **kwargs
                )
            except providers.ProviderError as e:
                last = e
                if meter is not None and e.usage is not None:
                    meter.book(model, role, e.usage, call_id=call_id, **tags)
                if not any(t in str(e) for t in _TRUNCATED):
                    raise
            except Exception as e:
                last = e
                name = type(e).__name__
                if not any(t in name or t in str(e) for t in _TRANSIENT):
                    raise
                break
            else:
                if meter is not None:
                    meter.book(model, role, result[1], call_id=call_id, **tags)
                return result
        time.sleep(2**attempt)
    raise last


def make_executor(model: str, meter: Meter, audit_path: str | Path | None = None, **tags):
    """The analysis agent's executor. Anthropic: explicit prompt caching via
    per-event blocks (each event is its own never-mutated block, cache_control
    on the system block and on the LAST context block -- a block whose text
    grows every call is never the same block twice and would never be read
    back). Others: plain call, auto-cached server-side."""
    audit = CallAudit(audit_path or meter.path.with_name("provider_calls.jsonl"))
    if providers.family(model) == "anthropic":

        def executor(system: str, context_blocks: list[str], current_turn: str) -> str:
            meter.check()
            system_blocks = (
                [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                if system
                else system
            )
            blocks = [{"type": "text", "text": t} for t in context_blocks]
            blocks[-1] = dict(blocks[-1], cache_control={"type": "ephemeral"})
            if current_turn:
                blocks.append({"type": "text", "text": current_turn})
            _log_prompt("exec", system, "".join(context_blocks) + current_turn, model=model, **tags)
            call_id = uuid.uuid4().hex
            try:
                obj, u = providers.complete_json_cached(
                    model,
                    system_blocks,
                    blocks,
                    agent._DECISION_SCHEMA,
                    max_tokens=32768,
                    expected_prompt="".join(context_blocks) + current_turn,
                    expected_system=system,
                    audit=audit,
                    call_id=call_id,
                    tags=tags,
                )
            except providers.ProviderError as error:
                if error.usage is not None:
                    meter.book(model, "exec", error.usage, call_id=call_id, **tags)
                raise
            meter.book(model, "exec", u, call_id=call_id, **tags)
            return json.dumps(obj)

        executor.wants_context_blocks = True  # type: ignore[attr-defined]
        return executor

    def executor(system: str, prompt: str) -> str:
        meter.check()
        _log_prompt("exec", system, prompt, model=model, **tags)
        obj, _usage = _call_plain(
            model,
            system,
            prompt,
            agent._DECISION_SCHEMA,
            32768,
            meter=meter,
            role="exec",
            audit=audit,
            tags=tags,
        )
        return json.dumps(obj)

    return executor


def _instrument(schema: dict) -> str:
    """Which judge instrument a call belongs to, by schema identity. Without
    this every judge call books under the condition name and the cost cannot
    be attributed -- 16% of calls carried 65% of output tokens in L1 and
    there was no way to say which instrument they were."""
    import branchpoint

    for name in ("detect", "review"):
        if schema is branchpoint.PROTOCOL[name]["schema"]:
            return name
    return "other"


def make_judge(model: str, meter: Meter, audit_path: str | Path | None = None, **tags):
    """branchpoint's judge_fn: one structured call, metered per instrument."""
    system = (
        "You are a careful methodological reviewer. Answer only with the requested JSON object."
    )

    audit = CallAudit(audit_path or meter.path.with_name("provider_calls.jsonl"))

    def call(prompt: str, schema: dict, **linkage) -> dict:
        meter.check()
        instrument = _instrument(schema)
        call_tags = {**tags, **linkage, "instrument": instrument}
        call_tags.setdefault("stage", instrument)
        _log_prompt("judge", system, prompt, model=model, **call_tags)
        obj, _usage = _call_plain(
            model,
            system,
            prompt,
            schema,
            16384,
            meter=meter,
            role="judge",
            audit=audit,
            tags=call_tags,
        )
        return obj

    def judge_fn(prompt: str, schema: dict) -> dict:
        return call(prompt, schema)

    judge_fn.model = model  # type: ignore[attr-defined]
    judge_fn.call = call  # type: ignore[attr-defined]
    return judge_fn
