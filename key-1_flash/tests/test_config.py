from __future__ import annotations

from agent.config.settings import Settings


def test_settings_loads_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODEL_ID", "local/test-model")
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_ENABLE_4BIT", "false")
    monkeypatch.setenv("AGENT_DEFAULT_MODE", "plan")
    monkeypatch.setenv("AGENT_DEFAULT_EFFORT", "high")

    settings = Settings.load()

    assert settings.model_id == "local/test-model"
    assert settings.cpu_model_id == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert settings.data_dir == tmp_path
    assert settings.enable_4bit is False
    assert settings.memory_dir == tmp_path / "memory"
    assert settings.default_mode == "plan"
    assert settings.default_effort == "high"
