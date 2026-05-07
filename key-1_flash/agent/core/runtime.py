"""Runtime controls for agent autonomy and reasoning budget."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EffortProfile:
    """Execution budget selected by the user at runtime."""

    name: str
    max_attempts: int
    description: str


EFFORT_PROFILES: dict[str, EffortProfile] = {
    "low": EffortProfile(
        name="low",
        max_attempts=1,
        description="one model pass; minimal reflection and no retries",
    ),
    "medium": EffortProfile(
        name="medium",
        max_attempts=3,
        description="up to three think/act/observe passes with focused retries",
    ),
    "high": EffortProfile(
        name="high",
        max_attempts=5,
        description="up to five think/act/observe passes for harder tasks",
    ),
}

VALID_MODES = {"plan", "build"}


@dataclass(slots=True)
class AgentRuntimeState:
    """Mutable terminal-session state that should not be persisted as config."""

    mode: str = "build"
    effort: str = "medium"

    @property
    def effort_profile(self) -> EffortProfile:
        return EFFORT_PROFILES[self.effort]

    def set_mode(self, mode: str) -> None:
        normalized = mode.strip().lower()
        if normalized not in VALID_MODES:
            raise ValueError("mode must be 'plan' or 'build'")
        self.mode = normalized

    def set_effort(self, effort: str) -> None:
        normalized = effort.strip().lower()
        if normalized not in EFFORT_PROFILES:
            raise ValueError("effort must be 'low', 'medium', or 'high'")
        self.effort = normalized

    def mode_summary(self) -> str:
        if self.mode == "plan":
            return "plan mode: tools are blocked; the agent can only think and propose steps"
        return "build mode: tools are available; the agent can inspect, edit, and execute"

    def effort_summary(self) -> str:
        profile = self.effort_profile
        return (
            f"{profile.name} effort: {profile.description}; "
            f"max attempts={profile.max_attempts}"
        )
