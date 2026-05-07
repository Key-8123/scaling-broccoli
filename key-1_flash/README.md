# Local AI Agent

A terminal-first autonomous AI assistant framework that runs a local Hugging Face instruction model, keeps persistent memory, and can call tools for files, shell commands, Python snippets, search, and file editing.

The default GPU model is `Qwen/Qwen2.5-Coder-7B-Instruct`, a capable 7B-class coding model. On CPU-only machines the app automatically falls back to `Qwen/Qwen2.5-Coder-0.5B-Instruct` so `python main.py` can still run locally. Models are downloaded automatically by `transformers` on first run and cached under `.agent_data/models`.

## Features

- Local Hugging Face model execution with `transformers`, `accelerate`, and `torch`
- GPU detection with CPU fallback
- Optional 4-bit quantization when `bitsandbytes` is installed and CUDA is available
- Rich terminal UI with startup banner, colored status, chat panels, memory status, tool status, and loading spinners
- Conversational chat with persistent history
- Short-term memory and long-term JSONL memory
- Persistent vector memory using FAISS when available, with a deterministic local fallback
- Plugin-ready tool registry
- Built-in tools for file read/write/edit, text search, shell execution, and Python execution
- THINK -> ACT -> OBSERVE -> REFLECT retry loop
- Runtime `/mode` control: `plan` blocks tools for thinking only, `build` enables agentic work
- Runtime `/effort` control: `low`, `medium`, and `high` adjust reflection/retry budget
- Rotating file logs under `.agent_data/logs/agent.log`
- Typed environment-driven configuration
- Pytest coverage for config, memory, tools, model wrapper, reflection, and chat orchestration

## Architecture

```text
agent/
  config/        Environment-backed settings
  core/          Agent orchestration, prompts, parser, reflection loop, logging
  memory/        Short-term, long-term, and vector memory
  models/        Local Hugging Face model wrapper
  tools/         Tool protocol, registry, filesystem, shell, Python execution
  ui/            Rich terminal interface
tests/           Automated test suite
main.py          Console entrypoint
```

The code keeps model inference, tool execution, memory, and UI separate so each area can evolve independently. New tools only need to implement the `Tool` protocol and register with `ToolRegistry`.

## Installation

Python 3.10+ is recommended. A CUDA-capable GPU is strongly recommended for 7B-class models.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

For CUDA 4-bit quantization, install a compatible `bitsandbytes` build for your platform. If it is unavailable, the agent runs without quantization.

## Usage

```bash
python main.py
```

Useful runtime commands:

- `/memory` shows short-term and long-term memory counts
- `/mode` shows the current autonomy mode
- `/mode plan` switches to thinking-only mode; no tools, file edits, shell commands, or Python execution
- `/mode build` switches to agentic mode; tools can inspect, edit, and execute locally
- `/effort` shows the active reasoning budget
- `/effort low|medium|high` changes the maximum THINK -> ACT -> OBSERVE -> REFLECT attempts
- `/exit` exits the session

Example `.env` overrides:

```env
AGENT_MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct
AGENT_CPU_MODEL_ID=Qwen/Qwen2.5-Coder-0.5B-Instruct
AGENT_FORCE_LARGE_MODEL=false
AGENT_DATA_DIR=.agent_data
AGENT_ENABLE_4BIT=true
AGENT_SHELL_TIMEOUT=30
AGENT_DEFAULT_MODE=build
AGENT_DEFAULT_EFFORT=medium
```

Alternative 6B/7B-class models:

```env
AGENT_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
```

```env
AGENT_MODEL_ID=deepseek-ai/deepseek-coder-6.7b-instruct
```

## Testing

```bash
pytest -q
```

The tests mock heavyweight model loading, so they do not download multi-gigabyte model weights. They verify the model wrapper, config loading, memory persistence, vector search fallback, tool execution, reflection retries, and chat orchestration.

## Operational Notes

- First model startup can take a long time because Hugging Face downloads model files.
- The 7B model requires substantial RAM or VRAM. CPU-only machines use the configured CPU fallback unless `AGENT_FORCE_LARGE_MODEL=true`.
- Shell and Python tools execute local code. Use this framework in a trusted environment.
- Use `/mode plan` when you want analysis only. Use `/mode build` when you want the agent to act.
- Logs are written to `.agent_data/logs/agent.log`.
- Persistent memory is stored under `.agent_data/memory`.

## Roadmap

- Add a safer command approval policy for shell and Python tools
- Add streaming token output in the Rich UI
- Add a plugin discovery directory
- Add richer planning objects and task state
- Add optional ChromaDB backend alongside FAISS
- Add benchmark scripts for model latency and memory retrieval quality
