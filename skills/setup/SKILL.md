---
name: setup
description: Set up and verify the nanogen development environment. Use when onboarding a new developer, after cloning the repo, when `uv sync` fails, when dependencies are broken, or when the user asks to install/fix/reset the environment. Triggers on "setup nanogen", "install dependencies", "uv sync failing", "fix environment", "fresh install", "verify setup".
---

# nanogen Setup

## Prerequisites

1. **Python >=3.10, <3.14**
2. **SSH access to G-REPA GitHub org** — private deps are fetched via SSH
3. **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup Steps

```sh
# 1. Install all dependencies (torch 2.10.0+cu128 + private G-REPA packages)
uv sync

# 2. Verify the environment
uv run python skills/setup/scripts/verify_setup.py
```

## Architecture

- **pyproject.toml** — single source of truth for all deps
- **uv.lock** — committed lockfile, resolves for Linux only
- **PyTorch** — pinned to `2.10.0+cu128`, sourced exclusively from `https://download.pytorch.org/whl/cu128`
- **Private deps** — `dpg-evaluator`, `geneval-evaluator`, `t2v-metrics` from G-REPA org via SSH, branch `nanogen/package`

## Troubleshooting

See [references/troubleshooting.md](references/troubleshooting.md) for fixes to common issues:
- 403/permission denied during git clone
- torch version resolution failures
- hardlink warnings
- nuclear reset (`rm -rf .venv uv.lock && uv sync`)

## Verification

Run `uv run python skills/setup/scripts/verify_setup.py` to check Python version, uv, SSH access, torch+CUDA, and private package imports.
