"""Parsing utilities for model tool-call responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParsedResponse:
    """Normalized model response."""

    thought: str
    final: str | None = None
    tool: str | None = None
    args: dict[str, Any] | None = None


def parse_model_response(text: str) -> ParsedResponse:
    """Parse the first JSON object in a model response, falling back to final text."""

    candidate = text.strip()
    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return ParsedResponse(thought="Unstructured response", final=text.strip())

    action = data.get("action")
    if isinstance(action, dict):
        tool = action.get("tool")
        args = action.get("args") or {}
        if not isinstance(args, dict):
            return ParsedResponse(
                thought=str(data.get("thought", "Malformed tool arguments")),
                final=(
                    "I could not safely execute that tool call because the model "
                    "returned invalid tool arguments. Please rephrase the request."
                ),
            )
        if not isinstance(tool, str) or not tool.strip():
            return ParsedResponse(
                thought=str(data.get("thought", "Malformed tool name")),
                final=str(data.get("final") or data.get("thought") or ""),
            )
        return ParsedResponse(
            thought=str(data.get("thought", "")),
            tool=tool,
            args=args,
        )
    return ParsedResponse(thought=str(data.get("thought", "")), final=str(data.get("final", "")))
