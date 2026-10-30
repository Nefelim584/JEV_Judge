"""Client for llama-server's OpenAI-compatible chat endpoint, with JSON-schema constrained output.

Blocking ``urllib`` calls: the runner runs one call per thread, with as many threads as the server
has slots (``llama-server -np N``), so no async stack is needed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class LLMError(RuntimeError):
    """Transport failure or an unusable response after all retries."""


@dataclass
class Reply:
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def inline_refs(schema: dict) -> dict:
    """Pydantic puts nested models under ``$defs``; inline them so every grammar converter accepts it."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def json_schema(model: type[BaseModel]) -> dict:
    return inline_refs(model.model_json_schema())


def parse_json(text: str) -> Any:
    """The grammar keeps the output to one JSON value; strip anything a template may put around it."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start : end + 1])


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 600, max_retries: int = 3,
                 response_format: str = "json_schema", extra_body: dict | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.response_format = response_format
        self.extra_body = extra_body or {}

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMClient":
        return cls(cfg["base_url"], cfg["model"], cfg.get("timeout_s", 600), cfg.get("max_retries", 3),
                   cfg.get("response_format", "json_schema"), cfg.get("extra_body"))

    def request_body(self, messages: list[dict], schema: dict | None, sampling: dict) -> dict:
        body = {"model": self.model, "messages": messages, **sampling, **self.extra_body}
        if schema is not None and self.response_format == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        return body

    def chat(self, body: dict) -> Reply:
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(60, 2**attempt))
            t0 = time.monotonic()
            try:
                req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    out = json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last = e
                continue
            choice = out["choices"][0]
            usage = out.get("usage", {})
            return Reply(choice["message"].get("content") or "", choice.get("finish_reason", ""),
                         usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), time.monotonic() - t0)
        raise LLMError(f"{self.url}: {last}")
