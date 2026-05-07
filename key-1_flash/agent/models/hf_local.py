"""Local Hugging Face causal language model wrapper."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import torch

from agent.config.settings import Settings

LOGGER = logging.getLogger(__name__)


class LocalModel:
    """Download, cache, and run a local instruction model."""

    def __init__(self, settings: Settings, ui: Any | None = None) -> None:
        self.settings = settings
        self.ui = ui
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = self._select_model_id()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    async def load(self) -> None:
        """Load the local model asynchronously."""

        await asyncio.to_thread(self._load_sync)

    def _select_model_id(self) -> str:
        if self.device == "cpu" and not self.settings.force_large_model:
            LOGGER.warning(
                "CUDA is unavailable; using CPU fallback model %s. Set "
                "AGENT_FORCE_LARGE_MODEL=true to force %s.",
                self.settings.cpu_model_id,
                self.settings.model_id,
            )
            return self.settings.cpu_model_id
        return self.settings.model_id

    def _load_sync(self) -> None:
        """Blocking model load isolated for async startup."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "cache_dir": str(self.settings.model_cache_dir),
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
            kwargs["dtype"] = torch.float16
            if self.settings.enable_4bit:
                try:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("4-bit quantization unavailable: %s", exc)
        else:
            kwargs["dtype"] = torch.float32

        LOGGER.info("Loading model %s on %s", self.model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=str(self.settings.model_cache_dir)
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except RuntimeError as exc:
            if self.device == "cpu" and "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "Model loading ran out of CPU memory. Use a smaller "
                    "AGENT_CPU_MODEL_ID, add a CUDA GPU, or set up quantized "
                    "inference with a supported backend."
                ) from exc
            raise
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from chat messages."""

        if not self.is_loaded:
            await self.load()
        return await asyncio.to_thread(self._generate_sync, messages)

    def _generate_sync(self, messages: list[dict[str, str]]) -> str:
        assert self.tokenizer is not None
        assert self.model is not None

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=self.settings.temperature > 0,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
