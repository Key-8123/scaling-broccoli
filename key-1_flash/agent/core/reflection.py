"""Think-act-observe-reflect loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.config.settings import Settings
from agent.core.parser import ParsedResponse, parse_model_response
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReflectionResult:
    """Final result of an agent reasoning loop."""

    final: str
    attempts: int
    tool_results: list[ToolResult]


class ReflectionLoop:
    """Coordinates model reasoning, tool execution, observation, and retry."""

    def __init__(self, settings: Settings, tools: ToolRegistry) -> None:
        self.settings = settings
        self.tools = tools

    async def run(
        self,
        model: object,
        messages: list[dict[str, str]],
        max_attempts: int | None = None,
    ) -> ReflectionResult:
        """Run the loop until a final answer or retry limit is reached."""

        tool_results: list[ToolResult] = []
        working_messages = list(messages)
        attempt_limit = max_attempts or self.settings.max_reflection_retries + 1
        for attempt in range(1, attempt_limit + 1):
            try:
                raw = await model.generate(working_messages)  # type: ignore[attr-defined]
                parsed = parse_model_response(raw)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("REFLECT model/parsing failure on attempt=%s", attempt)
                return ReflectionResult(
                    f"Model response handling failed: {exc}",
                    attempt,
                    tool_results,
                )
            LOGGER.info("THINK attempt=%s thought=%s", attempt, parsed.thought)

            if parsed.final is not None:
                LOGGER.info("REFLECT final produced on attempt=%s", attempt)
                return ReflectionResult(parsed.final, attempt, tool_results)

            if not self.settings.enable_tools or not parsed.tool:
                return ReflectionResult(raw, attempt, tool_results)

            result = await self.tools.run(parsed.tool, **(parsed.args or {}))
            tool_results.append(result)
            LOGGER.info(
                "ACT tool=%s ok=%s error=%s", parsed.tool, result.ok, result.error
            )
            observation = _format_observation(parsed, result)
            working_messages.append({"role": "assistant", "content": raw})
            working_messages.append({"role": "user", "content": observation})

            if result.ok:
                working_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Reflect on the observation and provide the final answer "
                            "as JSON with a final field."
                        ),
                    }
                )
            else:
                working_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The action failed. Reflect on the failure, retry with a "
                            "corrected action if possible, or provide a final answer."
                        ),
                    }
                )

        return ReflectionResult(
            "I could not complete the task within the configured retry limit.",
            attempt_limit,
            tool_results,
        )


def _format_observation(parsed: ParsedResponse, result: ToolResult) -> str:
    status = "success" if result.ok else "failure"
    body = result.output if result.ok else result.error or "unknown error"
    return f"OBSERVE tool={parsed.tool} status={status}\n{body}"
