"""Provider-neutral structured-output client for experiments.

One function, `complete_json(model, system, prompt, schema, max_tokens)`,
returning `(parsed_object, usage_dict)`. Provider is inferred from the model
id. The three families differ only in how a JSON schema is attached to the
request; nothing else about an occasion may vary, so the call shape is kept
identical across families (single user turn, system instruction, schema
enforced by the provider rather than by parsing prose).

"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from typing import Any

_anthropic_client = None
_openai_client = None
_openrouter_client = None


class ProviderError(RuntimeError):
    """Retryable failure from any provider, normalized."""

    def __init__(self, message: str, usage: dict[str, int] | None = None):
        super().__init__(message)
        self.usage = usage


AuditSink = Callable[[dict], None]


def _exception(error: Exception) -> dict:
    """Serialize useful provider failure fields without headers or credentials."""
    response = getattr(error, "response", None)
    body = getattr(error, "body", None)
    if body is None and response is not None:
        try:
            body = response.json()
        except Exception:
            body = None
    return {
        "type": type(error).__name__,
        "message": str(error),
        "status_code": getattr(error, "status_code", None),
        "request_id": getattr(error, "request_id", None),
        "code": getattr(error, "code", None),
        "param": getattr(error, "param", None),
        "body": body,
    }


def _send(
    provider: str,
    model: str,
    request: dict,
    send: Callable[[], Any],
    audit: AuditSink | None,
    call_id: str,
    tags: dict,
) -> tuple[Any, dict]:
    if audit:
        audit({**tags, "kind": "request", "call_id": call_id, "provider": provider,
               "model": model, "request": request})
    try:
        response = send()
    except Exception as error:
        if audit:
            audit({**tags, "kind": "response", "call_id": call_id, "provider": provider,
                   "model": model, "response": None, "error": _exception(error)})
        raise
    raw = response.model_dump(mode="json")
    if audit:
        audit({**tags, "kind": "response", "call_id": call_id, "provider": provider,
               "model": model, "response": raw, "error": None})
    return response, raw


def family(model: str) -> str:
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o3", "o4")):
        return "openai"
    raise ValueError(f"unknown model family for {model!r}")


def _openrouter(model, system, prompt, schema, max_tokens, audit, call_id, tags):
    """OpenRouter (OpenAI-compatible chat completions). Model ids carry the
    'openrouter/' prefix to select this family, e.g.
    'openrouter/z-ai/glm-4.6'; the prefix is stripped for the request.
    Prompt caching on OpenRouter is provider-side and automatic where the
    upstream supports it -- cached_tokens are surfaced when reported."""
    global _openrouter_client
    from openai import OpenAI

    if _openrouter_client is None:
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
        )
    # require_parameters: route ONLY to providers that actually enforce the
    # JSON schema. Without it OpenRouter may land the call on a provider
    # that treats response_format as a hint and returns unvalidated text.
    # OPENROUTER_PROVIDERS pins an explicit order (fallbacks off) for models
    # whose schema-capable endpoints differ in health.
    prov: dict = {"require_parameters": True}
    if os.environ.get("OPENROUTER_PROVIDERS"):
        prov["order"] = os.environ["OPENROUTER_PROVIDERS"].split(",")
        prov["allow_fallbacks"] = False
    extra: dict = {"provider": prov}
    effort = os.environ.get("OPENROUTER_REASONING_EFFORT")
    if effort:  # some models default reasoning to max and bill it as output
        extra["reasoning"] = {"effort": effort}
    request = {
        "model": model[len("openrouter/") :],
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "decision", "schema": schema, "strict": True},
        },
        "extra_body": extra,
    }
    response, _raw = _send(
        "openrouter", model, request,
        lambda: _openrouter_client.chat.completions.create(**request),
        audit, call_id, tags,
    )
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "cached_input_tokens": getattr(
            getattr(response.usage, "prompt_tokens_details", None), "cached_tokens", 0
        ) or 0,
    }
    choice = response.choices[0]
    if choice.finish_reason not in ("stop", None):
        raise ProviderError(f"openrouter finish_reason {choice.finish_reason!r}", usage)
    text = choice.message.content
    if not text:
        raise ProviderError("openrouter returned empty content", usage)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        if "Extra data" not in str(e):
            raise ProviderError(f"openrouter returned non-JSON: {e}", usage) from e
        parsed, _ = json.JSONDecoder().raw_decode(text.lstrip())
    return parsed, usage


def _anthropic(model, system, prompt, schema, max_tokens, audit, call_id, tags):
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        _anthropic_client = anthropic.Anthropic(default_headers=(
            {"anthropic-workspace-id": workspace} if workspace else None))
    # Streaming: the SDK refuses non-streaming calls whose max_tokens
    # implies a >10-min worst case (seen at 16384 on the brief call).
    output_config = {"format": {"type": "json_schema", "schema": schema}}
    if effort := os.environ.get("ANTHROPIC_REASONING_EFFORT"):
        output_config["effort"] = effort
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "output_config": output_config,
        "messages": [{"role": "user", "content": prompt}],
    }

    def send():
        with _anthropic_client.messages.stream(**request) as stream:
            return stream.get_final_message()

    response, _raw = _send("anthropic", model, request, send, audit, call_id, tags)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0
        ) or 0,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    }
    if response.stop_reason not in ("end_turn", "stop_sequence"):
        raise ProviderError(f"anthropic stopped with {response.stop_reason!r}", usage)
    text = next(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(text), usage
    except json.JSONDecodeError as error:
        raise ProviderError(f"anthropic returned non-JSON: {error}", usage) from error


def _openai(model, system, prompt, schema, max_tokens, audit, call_id, tags):
    global _openai_client
    from openai import OpenAI

    if _openai_client is None:
        _openai_client = OpenAI()
    effort = os.environ.get("OPENAI_REASONING_EFFORT")
    request = {
        "model": model,
        "instructions": system,
        "input": prompt,
        "max_output_tokens": max_tokens,
        **({"reasoning": {"effort": effort}} if effort else {}),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "enumeration",
                "schema": schema,
                "strict": True,
            }
        },
    }
    response, _raw = _send(
        "openai", model, request, lambda: _openai_client.responses.create(**request),
        audit, call_id, tags,
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cached_input_tokens": getattr(
            getattr(response.usage, "input_tokens_details", None), "cached_tokens", 0
        ) or 0,
    }
    if response.status != "completed":
        raise ProviderError(
            f"openai status {response.status!r} "
            f"details={getattr(response, 'incomplete_details', None)!r}",
            usage,
        )
    try:
        parsed = json.loads(response.output_text)
    except json.JSONDecodeError as e:
        if "Extra data" not in str(e):
            raise ProviderError(f"openai returned non-JSON: {e}", usage) from e
        # Some models append junk after a valid strict-schema object
        # (seen live: gpt-5.6-luna, AG stage 1) -- take the leading object.
        parsed, _ = json.JSONDecoder().raw_decode(response.output_text.lstrip())
    # OpenAI auto-caches prefixes >=1024 tokens server-side; surface the
    # cached share so callers bill it at the cached rate instead of full.
    return parsed, usage


_DISPATCH = {
    "anthropic": _anthropic,
    "openai": _openai,
    "openrouter": _openrouter,
}


def complete_json(
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    max_tokens: int = 8192,
    *,
    audit: AuditSink | None = None,
    call_id: str | None = None,
    tags: dict | None = None,
) -> tuple[Any, dict[str, int]]:
    """Return (parsed JSON object, usage). Raises ProviderError on a
    retryable provider-side failure; anything else propagates."""
    return _DISPATCH[family(model)](
        model, system, prompt, schema, max_tokens, audit, call_id or uuid.uuid4().hex, tags or {}
    )


def complete_json_cached(
    model: str,
    system: str | list[dict],
    content_blocks: list[dict],
    schema: dict[str, Any],
    max_tokens: int = 8192,
    *,
    expected_prompt: str,
    expected_system: str | None = None,
    audit: AuditSink | None = None,
    call_id: str | None = None,
    tags: dict | None = None,
) -> tuple[Any, dict[str, int]]:
    """Call Anthropic with cache annotations that preserve the plain prompt bytes."""
    if family(model) != "anthropic":
        raise ValueError(f"complete_json_cached is Anthropic-only, got {model!r}")

    reconstructed = "".join(b.get("text", "") for b in content_blocks)
    if reconstructed != expected_prompt:
        raise ValueError("cache annotations changed the plain prompt")
    if expected_system is not None and isinstance(system, list):
        reconstructed_system = "".join(b.get("text", "") for b in system)
        if reconstructed_system != expected_system:
            raise ValueError("cache annotations changed the system prompt")

    return _anthropic(
        model, system, content_blocks, schema, max_tokens,
        audit, call_id or uuid.uuid4().hex, tags or {},
    )


def native_request(
    model: str,
    system: str,
    history: list[dict],
    client: dict | list[dict],
    tools: list[dict] | None = None,
    require_tool: bool = False,
) -> dict:
    """Build the exact secret-free payload for one auditable agent turn."""
    provider = family(model)
    def item(value: dict) -> dict:
        if value["kind"] == "text":
            return {"role": "user", "content": value["content"]}
        if provider == "openai":
            return {
                "type": "function_call_output",
                "call_id": value["call_id"],
                "output": value["content"],
            }
        if provider == "anthropic":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": value["call_id"],
                        "content": value["content"],
                    }
                ],
            }
        return {
            "role": "tool",
            "tool_call_id": value["call_id"],
            "content": value["content"],
        }

    values = client if isinstance(client, list) else [client]
    items = [item(value) for value in values]
    if provider == "anthropic" and len(items) > 1:
        items = [
            {
                "role": "user",
                "content": [
                    block
                    for message in items
                    for block in (
                        message["content"]
                        if isinstance(message["content"], list)
                        else [{"type": "text", "text": message["content"]}]
                    )
                ],
            }
        ]

    if provider == "openai":
        request = {
            "model": model,
            "instructions": system,
            "input": [*history, *items],
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "max_output_tokens": 32768,
            "reasoning": {
                "context": "all_turns",
                "summary": "auto",
                **(
                    {"effort": os.environ["OPENAI_REASONING_EFFORT"]}
                    if os.environ.get("OPENAI_REASONING_EFFORT")
                    else {}
                ),
            },
        }
        if tools:
            request.update(
                tools=[{"type": "function", **tool, "strict": True} for tool in tools],
                tool_choice="required" if require_tool else "none",
                parallel_tool_calls=False,
            )
        return request

    if provider == "anthropic":
        if model not in {"claude-opus-5", "claude-sonnet-5"}:
            raise ValueError(f"continuation supports adaptive-thinking Claude 5 only, got {model!r}")
        request = {
            "model": model,
            "system": system,
            "messages": [*history, *items],
            "max_tokens": 32768,
            "thinking": {"type": "adaptive", "display": "summarized"},
            **(
                {"output_config": {"effort": os.environ["ANTHROPIC_REASONING_EFFORT"]}}
                if os.environ.get("ANTHROPIC_REASONING_EFFORT")
                else {}
            ),
        }
        if tools:
            request.update(
                tools=[
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": tool["parameters"],
                        "strict": True,
                    }
                    for tool in tools
                ],
                tool_choice={
                    "type": "any" if require_tool else "none",
                    **({"disable_parallel_tool_use": True} if require_tool else {}),
                },
            )
        return request

    providers = os.environ.get("OPENROUTER_PROVIDERS")
    if not providers or "," in providers:
        raise ValueError("continuation requires exactly one pinned OPENROUTER_PROVIDERS entry")
    messages = history or ([{"role": "system", "content": system}] if system else [])
    request = {
        "model": model.removeprefix("openrouter/"),
        "messages": [*messages, *items],
        "max_tokens": 32768,
        "provider": {
            "require_parameters": True,
            "order": providers.split(","),
            "allow_fallbacks": False,
        },
    }
    if effort := os.environ.get("OPENROUTER_REASONING_EFFORT"):
        request["reasoning"] = {"effort": effort}
    if tools:
        request.update(
            tools=[
                {
                    "type": "function",
                    "function": {**tool, "strict": True},
                }
                for tool in tools
            ],
            tool_choice="auto" if require_tool else "none",
        )
    return request


def native_complete(model: str, request: dict) -> dict:
    """Send a prepared payload and return the provider's complete raw object."""
    global _anthropic_client, _openai_client, _openrouter_client
    provider = family(model)
    if provider == "openai":
        from openai import OpenAI

        _openai_client = _openai_client or OpenAI()
        response = _openai_client.responses.create(**request)
    elif provider == "anthropic":
        import anthropic

        if _anthropic_client is None:
            workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": workspace} if workspace else None
            _anthropic_client = anthropic.Anthropic(default_headers=headers)
        with _anthropic_client.messages.stream(**request) as stream:
            response = stream.get_final_message()
    else:
        from openai import OpenAI

        _openrouter_client = _openrouter_client or OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
        )
        body = {key: value for key, value in request.items() if key not in {"provider", "reasoning"}}
        response = _openrouter_client.chat.completions.create(
            **body,
            extra_body={key: request[key] for key in ("provider", "reasoning") if key in request},
        )
    return response.model_dump(mode="json")


def native_history(model: str, request: dict, response: dict) -> list[dict]:
    """Return the complete provider-native prefix, preserving opaque reasoning."""
    provider = family(model)
    if provider == "openai":
        output = [
            {
                key: value
                for key, value in item.items()
                if value is not None and not (item["type"] == "reasoning" and key == "status")
            }
            for item in response["output"]
        ]
        return [*request["input"], *output]
    if provider == "anthropic":
        content = [
            {key: value for key, value in block.items() if value is not None and key != "parsed_output"}
            for block in response["content"]
        ]
        return [*request["messages"], {"role": "assistant", "content": content}]
    message = {
        key: value
        for key, value in response["choices"][0]["message"].items()
        if value is not None
    }
    return [*request["messages"], message]


def native_result(model: str, response: dict) -> tuple[str, list[dict], dict[str, int]]:
    """Normalize visible text, client tool calls, and usage without losing raw data."""
    provider = family(model)
    if provider == "openai":
        if response.get("status") != "completed":
            raise ProviderError(f"openai status {response.get('status')!r}")
        text = "".join(
            block.get("text", "")
            for item in response["output"]
            if item["type"] == "message"
            for block in item.get("content", [])
            if block["type"] == "output_text"
        )
        calls = [
            {
                "id": item["call_id"],
                "name": item["name"],
                "arguments": json.loads(item["arguments"]),
            }
            for item in response["output"]
            if item["type"] == "function_call"
        ]
    elif provider == "anthropic":
        if response.get("stop_reason") not in {"end_turn", "tool_use", "stop_sequence"}:
            raise ProviderError(f"anthropic stopped with {response.get('stop_reason')!r}")
        text = "".join(block.get("text", "") for block in response["content"] if block["type"] == "text")
        calls = [
            {"id": block["id"], "name": block["name"], "arguments": block["input"]}
            for block in response["content"]
            if block["type"] == "tool_use"
        ]
    else:
        choice = response["choices"][0]
        if choice.get("finish_reason") not in {"stop", "tool_calls", None}:
            raise ProviderError(f"openrouter finish_reason {choice.get('finish_reason')!r}")
        message = choice["message"]
        text = message.get("content") or ""
        calls = [
            {
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
            for call in (message.get("tool_calls") or [])
        ]
    return text, calls, native_usage(model, response)


def native_usage(model: str, response: dict) -> dict[str, int]:
    """Read billable usage even when the provider response is rejected."""
    provider = family(model)
    raw = response["usage"]
    if provider == "openai":
        return {
            "input_tokens": raw["input_tokens"],
            "output_tokens": raw["output_tokens"],
            "cached_input_tokens": raw.get("input_tokens_details", {}).get("cached_tokens", 0),
        }
    if provider == "anthropic":
        return {
            "input_tokens": raw["input_tokens"],
            "output_tokens": raw["output_tokens"],
            "cache_creation_input_tokens": raw.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": raw.get("cache_read_input_tokens", 0),
        }
    return {
        "input_tokens": raw["prompt_tokens"],
        "output_tokens": raw["completion_tokens"],
        "cached_input_tokens": raw.get("prompt_tokens_details", {}).get("cached_tokens", 0),
    }
