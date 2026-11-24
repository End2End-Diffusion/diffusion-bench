# Setup Troubleshooting

## git clone fails during `uv sync`

Evaluator packages are fetched over HTTPS from the public `End2End-Diffusion/diffusionbench-evals` repo.

**Diagnosis:** `git ls-remote https://github.com/End2End-Diffusion/diffusionbench-evals.git`

**Fixes:**
- Check network / proxy access to github.com
- Check for a global git rewrite forcing SSH (`git config --global --get-regexp insteadof`) and remove it if you have no SSH key set up

## `No solution found` / torch version errors

Project pins `torch==2.10.0+cu128` from the PyTorch cu128 index.

**Common causes:**
- Wrong Python: must be >=3.10, <3.14
- Stale lockfile: `rm uv.lock && uv sync`
- Wrong platform: lockfile resolves for Linux only (`sys_platform == 'linux'` in pyproject.toml)
- Version mismatch: dependency must be `torch==2.10.0+cu128` (with `+cu128` suffix), not `torch==2.10.0`

## `Failed to hardlink files` warning during `uv sync`

Harmless — happens when `.venv` and uv cache are on different filesystems. Suppress with:
```sh
export UV_LINK_MODE=copy
```

## Nuclear reset

```sh
rm -rf .venv uv.lock && uv sync
```

## Evaluator import errors after install

If an evaluator package (e.g. `dpg_evaluator`, `geneval_evaluator`, `t2v_metrics`) fails to import:
1. Check it appears in `uv pip list | grep -iE "evaluator|t2v"`
2. Re-run `uv sync` and watch for git errors in output
