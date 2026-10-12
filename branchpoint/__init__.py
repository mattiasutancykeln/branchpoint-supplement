"""Explore consequential analysis decisions through reproducible forks."""

import hashlib
import json
import os
import random
import re
import shutil
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

__all__ = ["Project", "sample"]

PROTOCOL = tomllib.loads(Path(__file__).with_name("protocol.toml").read_text())
STATE = "bp_state.json"
FORKS = "forks/bp"
CALLS = ".branchpoint/calls.jsonl"
_CONTROL = {
    ".git",
    "__pycache__",
    "snapshots",
    "forks",
    STATE,
    "report",
}


def _relative(root: Path, path: Path, label: str) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} outside project") from error


class Project:
    def __init__(self, root: str | Path, segment: str = ""):
        self.root = Path(root).resolve()
        relative = _relative(self.root, self.root / segment, f"segment {segment}")
        self.path = self.root / relative
        self.segment = "" if relative == Path(".") else str(relative)

    def events(self) -> list[dict]:
        trace = self.path / "trace.jsonl"
        if not trace.exists():
            return []
        events = [json.loads(line) for line in trace.read_text().splitlines() if line]
        if any(event.get("i") != i for i, event in enumerate(events)):
            raise ValueError(f"non-contiguous trace: {trace}")
        return events

    def report(self) -> dict | None:
        reports = (event for event in reversed(self.events()) if event["kind"] == "report")
        return json.loads(event["content"]) if (event := next(reports, None)) else None

    def add(
        self,
        kind: str,
        content: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        **metadata,
    ) -> dict:
        trace = self.path / "trace.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "i": len(self.events()),
            "kind": kind,
            "content": content,
            "inputs": inputs or [],
            "outputs": outputs or [],
            **metadata,
        }
        with trace.open("a") as file:
            file.write(json.dumps(event) + "\n")
            file.flush()
            os.fsync(file.fileno())
        return event

    def _fork(self, boundary: int, name: str) -> "Project":
        events = self.events()
        if boundary not in range(len(events) + 1):
            raise ValueError(f"unknown boundary: {boundary}")

        def outputs(selected: list[dict]) -> set[Path]:
            paths = set()
            for output in (path for event in selected for path in event.get("outputs", [])):
                path = Path(output) if Path(output).is_absolute() else self.path / output
                try:
                    paths.add(path.resolve().relative_to(self.path.resolve()))
                except ValueError as error:
                    raise ValueError(f"output outside project: {output}") from error
            return paths

        before = events[:boundary]
        removed = outputs(events[boundary:])
        overlap = outputs(before) & removed
        restored = {}
        for event in before:
            if event["kind"] != "action":
                continue
            try:
                arguments = json.loads(event["content"])["arguments"]
                path = Path(arguments["filename"])
                script = arguments["script"]
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if path in overlap and path.parts[:1] == ("scripts",) and isinstance(script, str):
                restored[path] = script
        if unrecoverable := overlap - restored.keys():
            raise ValueError(f"overwritten before fork: {unrecoverable}")

        fork_root = (self.root / FORKS).resolve()
        relative = _relative(fork_root, fork_root / name, f"fork {name}")
        if relative == Path("."):
            raise ValueError(f"invalid fork name: {name}")
        target = fork_root / relative

        def ignored(directory: str, names: list[str]) -> set[str]:
            directory = Path(directory).resolve()
            skipped = {"__pycache__"} & set(names)
            if directory == self.path:
                skipped |= _CONTROL & set(names)
            if directory == self.path / ".branchpoint":
                skipped.add("provider_calls.jsonl")
            return skipped

        shutil.copytree(self.path, target, ignore=ignored)
        trace = "".join(json.dumps(event) + "\n" for event in before)
        (target / "trace.jsonl").write_text(trace)
        calls = target / CALLS
        if calls.exists():
            kept = [
                line
                for line in calls.read_text().splitlines()
                if json.loads(line)["event_i"] < boundary
            ]
            calls.write_text("".join(line + "\n" for line in kept))
        for path in removed:
            output = target / path
            if output.is_symlink() or output.is_file():
                output.unlink()
            elif output.is_dir():
                shutil.rmtree(output)
        for path, script in restored.items():
            output = target / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(script)
        return Project(self.root, str(target.relative_to(self.root)))

    def fork(self, step: int, name: str) -> "Project":
        """Fork immediately before an existing event."""
        if step not in {event["i"] for event in self.events()}:
            raise ValueError(f"unknown event: {step}")
        return self._fork(step, name)

    def fork_after(self, step: int, name: str) -> "Project":
        """Fork immediately after a completed event."""
        if step not in {event["i"] for event in self.events()}:
            raise ValueError(f"unknown event: {step}")
        return self._fork(step + 1, name)


def _ask(kind: str, judge: Callable, linkage: dict | None = None, **values) -> dict:
    spec = PROTOCOL[kind]
    prompt = spec["prompt"].format(**values)
    call = getattr(judge, "call", None)
    return (
        call(prompt, spec["schema"], stage=kind, **(linkage or {}))
        if call
        else judge(prompt, spec["schema"])
    )


def _clean(text: str) -> str:
    text = re.sub(r"^(do not|don't|never|avoid)\s+", "", text.strip(), flags=re.I)
    return re.split(r"\s+(instead of|rather than|in favor of|in place of)\s+", text, flags=re.I)[
        0
    ].rstrip(".")


def _tuple_tree(value):
    return tuple(map(_tuple_tree, value)) if isinstance(value, list) else value


def sample(
    root: str | Path,
    continue_fn: Callable,
    judge_fn: Callable,
    stop_fn: Callable = lambda: None,
    **overrides,
) -> dict:
    root = Path(root)
    defaults = dict(PROTOCOL["sampler"])
    configured = {}
    if (config := root / "branchpoint.toml").exists():
        configured = tomllib.loads(config.read_text()).get("sampler", {})
    if unknown := (configured.keys() | overrides.keys()) - defaults.keys():
        raise ValueError(f"unknown sampler options: {', '.join(sorted(unknown))}")
    options = defaults | configured | overrides
    for key in ("cap", "strikes", "workers"):
        if not isinstance(options[key], int) or isinstance(options[key], bool) or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if options["allocation"] not in {"uniform", "spread"}:
        raise ValueError(f"unknown allocation: {options['allocation']}")
    method = options["branching_method"]
    if method not in {"replay", "continuation"}:
        raise ValueError(f"unknown branching method: {method}")

    saved = json.loads((root / STATE).read_text()) if (root / STATE).exists() else {}
    registry = saved.get("registry", {})
    errors = saved.get("errors", [])
    attempts = saved.get("n_branches", 0)
    in_flight = saved.get("in_flight", {})
    probes = saved.get("probes", {})
    rng = random.Random(options["seed"])
    if state := saved.get("rng_state"):
        rng.setstate(_tuple_tree(state))
    identity = {
        key: options.get(key) for key in ("seed", "allocation", "strikes", "branching_method")
    }
    if options["allocation"] == "spread":
        identity["allocation_rule"] = "least-tried-occurrence-v1"
    saved_identity = saved.get("options")
    if saved_identity and saved_identity.get("allocation") == "uniform":
        saved_identity = {key: value for key, value in saved_identity.items() if key != "optimism"}
    if saved_identity not in (None, identity):
        raise ValueError("cannot resume with different sampler options")

    def discover(project: Project) -> None:
        events = project.events()
        interventions = [event["i"] for event in events if event["kind"] == "intervention"]
        start = interventions[-1] + 1 if interventions else 0
        kinds = {"llm"} if method == "replay" else {"analysis"}
        transcript = json.dumps(
            [event for event in events[start:] if event["kind"] in kinds], indent=2
        )
        decisions = _ask(
            "detect",
            judge_fn,
            linkage={"segment": project.segment or "root"},
            question=(root / "QUESTION.md").read_text(),
            transcript=transcript,
        )["decisions"]
        for decision in decisions:
            occurrence = decision["event_i"]
            if occurrence not in range(len(events)) or events[occurrence]["kind"] not in kinds:
                raise ValueError(f"detector returned invalid event {occurrence}")
        for decision in decisions:
            occurrence = decision["event_i"]
            key_text = f"{project.segment}:{occurrence}:{decision['decision']}"
            key = hashlib.sha1(key_text.encode()).hexdigest()[:10]
            decision["approach"] = _clean(decision["approach"])
            registry.setdefault(
                key,
                {
                    **decision,
                    "occurrence_i": occurrence,
                    "boundary_i": occurrence,
                    "boundary": (
                        "after_completed_response" if method == "continuation" else "before_event"
                    ),
                    "home": project.segment,
                    "bans": [decision["approach"]],
                    "branches": [],
                    "strikes": 0,
                    "closed": False,
                },
            )

    def continue_after(fork: Project):
        stop_fn()
        if fork.report() is None:
            return continue_fn(fork.path)
        return None

    def failure(fork: Project, error: Exception) -> str:
        message = f"{fork.segment}: {type(error).__name__}: {error}"
        errors.append(message)
        return message

    def save(stop_reason: str | None = None) -> dict:
        state = {
            "registry": registry,
            "n_branches": attempts,
            "errors": errors,
            "rng_state": rng.getstate(),
            "in_flight": in_flight,
            "probes": probes,
            "options": identity,
            "decisions": len(registry),
            "branching_method": method,
            "open_at_stop": sorted(key for key, item in registry.items() if not item["closed"]),
        }
        if stop_reason:
            state["stop_reason"] = stop_reason
        path = root / STATE
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)
        return state

    if not registry:
        discover(Project(root))

    def draw(opened: list[dict], count: int) -> list[dict]:
        if options.get("allocation", "uniform") != "spread":
            return rng.sample(opened, count)
        remaining = list(opened)
        rng.shuffle(remaining)
        remaining.sort(key=lambda decision: len(decision["branches"]) + decision["strikes"])
        return remaining[:count]

    reason = None
    with ThreadPoolExecutor(max_workers=options["workers"]) as pool:
        jobs = {
            pool.submit(continue_after, Project(root, segment)): (
                registry[key],
                Project(root, segment),
            )
            for segment, key in list(in_flight.items())[: options["workers"]]
        }
        while jobs or attempts < options["cap"]:
            while len(jobs) < options["workers"] and attempts < options["cap"]:
                active = set(in_flight.values())
                opened = [
                    decision
                    for key, decision in registry.items()
                    if not decision["closed"] and key not in active
                ]
                if not opened:
                    break
                batch = draw(
                    opened,
                    min(
                        len(opened),
                        options["workers"] - len(jobs),
                        options["cap"] - attempts,
                    ),
                )
                for decision in batch:
                    digest = hashlib.sha1(decision["decision"].encode()).hexdigest()[:8]
                    home = Project(root, decision["home"])
                    name = f"{digest}-{attempts}"
                    fork = (
                        home.fork_after(decision["boundary_i"], name)
                        if method == "continuation"
                        else home.fork(decision["boundary_i"], name)
                    )
                    note = PROTOCOL["intervention"]["note"].format(
                        decision=decision["decision"], bans=json.dumps(decision["bans"])
                    )
                    fork.add("intervention", note, role="user")
                    key = next(key for key, value in registry.items() if value is decision)
                    in_flight[fork.segment] = key
                    probes[fork.segment] = {
                        "decision_key": key,
                        "source_segment": decision["home"],
                        "boundary_i": decision["boundary_i"],
                        "boundary": decision["boundary"],
                        "intervention": note,
                        "status": "in_flight",
                    }
                    attempts += 1
                    jobs[pool.submit(continue_after, fork)] = (decision, fork)
                    save()

            if not jobs:
                break
            job = next(as_completed(jobs))
            decision, fork = jobs.pop(job)
            try:
                job.result()
                if not (report := fork.report()):
                    raise ValueError("continuation produced no report")
                events = fork.events()
                intervention = max(
                    i for i, event in enumerate(events) if event["kind"] == "intervention"
                )
                verdict = _ask(
                    "review",
                    judge_fn,
                    linkage={"segment": fork.segment},
                    decision=decision["decision"],
                    bans=json.dumps(decision["bans"]),
                    transcript=json.dumps(events[intervention + 1 :], indent=2),
                    note=report.get("note") or "",
                )
                decision["bans"].append(_clean(verdict["approach"]))
                if verdict["novel"]:
                    decision["branches"].append(fork.segment)
                    probes[fork.segment].update(status="retained", verdict=verdict)
                    try:
                        discover(fork)
                    except Exception as error:
                        probes[fork.segment]["discovery_error"] = failure(fork, error)
                else:
                    decision["strikes"] += 1
                    decision["closed"] = decision["strikes"] >= options["strikes"]
                    probes[fork.segment].update(status="rejected", verdict=verdict)
                in_flight.pop(fork.segment, None)
            except SystemExit:
                reason = "budget-guard"
            except Exception as error:
                message = failure(fork, error)
                probes[fork.segment].update(status="error", error=message)
                in_flight.pop(fork.segment, None)
            save()
            if reason:
                break

    reason = reason or ("cap-reached" if attempts >= options["cap"] else "pool-exhausted")
    return save(reason)
