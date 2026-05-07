from __future__ import annotations

import sys
import types

import torch

from agent.models.hf_local import LocalModel


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "user: hi\nassistant:"

    def __call__(self, prompt, return_tensors):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tokens, skip_special_tokens=True):
        return "ok"


class FakeModel:
    device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return None

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


def test_local_model_load_and_generate(monkeypatch, settings):
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeTokenizer()
        ),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeModel()
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    settings.model_id = "fake/model"

    model = LocalModel(settings)
    model._load_sync()

    assert model.is_loaded
    assert model._generate_sync([{"role": "user", "content": "hi"}]) == "ok"


def test_local_model_uses_cpu_fallback(settings):
    settings.model_id = "large/model"
    settings.cpu_model_id = "small/model"
    settings.force_large_model = False

    model = LocalModel(settings)

    if model.device == "cpu":
        assert model.model_id == "small/model"
    else:
        assert model.model_id == "large/model"
