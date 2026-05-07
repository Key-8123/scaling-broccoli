from __future__ import annotations

from pathlib import Path

import pytest

from agent.config.settings import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        shell_timeout=5,
        max_reflection_retries=1,
        short_term_limit=4,
        embedding_model="hash",
        max_new_tokens=8,
    )
