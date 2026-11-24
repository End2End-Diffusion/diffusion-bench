---
name: setup
description: Set up and verify the diffusion-bench development environment. Use when onboarding a new developer, after cloning the repo, when `uv sync` fails, when dependencies are broken, or when the user asks to install/fix/reset the environment. Triggers on "setup diffusion-bench", "install dependencies", "uv sync failing", "fix environment", "fresh install", "verify setup".
---

# diffusion-bench Setup

## Prerequisites

1. **Python >=3.10, <3.14**
2. **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup Steps

```sh
# 1. Install all dependencies (torch 2.10.0+cu128 + evaluator packages)
uv sync

# 2. Verify the environment
uv run python skills/setup/scripts/verify_setup.py
```

## Architecture

- **pyproject.toml** — single source of truth for all deps
- **uv.lock** — committed lockfile, resolves for Linux only
- **PyTorch** — `torch` and `torchvision` pinned to `+cu128`, sourced exclusively from `https://download.pytorch.org/whl/cu128`
- **Evaluators** — `dpg-evaluator`, `geneval-evaluator`, `geneval2-evaluator`, `prism-bench-evaluator`, `qwen-image-bench-evaluator`, `t2v-metrics` from [End2End-Diffusion/diffusionbench-evals](https://github.com/End2End-Diffusion/diffusionbench-evals) (public, HTTPS, branch `main`)

## Troubleshooting

See [references/troubleshooting.md](references/troubleshooting.md) for fixes to common issues:
- git clone failures during `uv sync`
- torch version resolution failures
- hardlink warnings
- nuclear reset (`rm -rf .venv uv.lock && uv sync`)

## Verification

Run `uv run python skills/setup/scripts/verify_setup.py` to check Python version, uv, torch+CUDA, and evaluator package imports.
