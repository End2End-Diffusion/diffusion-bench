# Setup Troubleshooting

## `403` / `Permission denied` on git clone during `uv sync`

Private G-REPA deps use SSH URLs (`ssh://git@github.com/G-REPA/...`).

**Diagnosis:** `ssh -T git@github.com` — should print "Hi <username>!"

**Fixes:**
- Add SSH key to GitHub account with G-REPA org access
- If only HTTPS token works, set global git rewrite:
  ```sh
  git config --global url."git@github.com:".insteadOf "https://github.com/"
  ```

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

## Private package import errors after install

If `dpg_evaluator`, `geneval_evaluator`, or `t2v_metrics` fail to import:
1. Check they appear in `uv pip list | grep -E "dpg|geneval|t2v"`
2. If missing, SSH access likely failed silently — check `ssh -T git@github.com`
3. Re-run `uv sync` and watch for git errors in output
