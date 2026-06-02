#!/usr/bin/env python3
"""Verify nanogen environment setup."""

import subprocess
import sys
import shutil


def check(label, ok, fix=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok and fix:
        print(f"         Fix: {fix}")
    return ok


def main():
    print("nanogen setup verification\n")
    all_ok = True

    # 1. Python version
    v = sys.version_info
    ok = v >= (3, 10) and v < (3, 14)
    all_ok &= check(
        f"Python {v.major}.{v.minor}.{v.micro} (need >=3.10, <3.14)", ok,
        "Install Python 3.10-3.13"
    )

    # 2. uv installed
    ok = shutil.which("uv") is not None
    all_ok &= check(
        "uv package manager", ok,
        "curl -LsSf https://astral.sh/uv/install.sh | sh"
    )

    # 3. SSH access to GitHub
    result = subprocess.run(
        ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "git@github.com"],
        capture_output=True, text=True, timeout=10
    )
    ssh_out = result.stdout + result.stderr
    ok = "successfully authenticated" in ssh_out.lower()
    all_ok &= check(
        "SSH access to GitHub", ok,
        "Add SSH key to GitHub: https://github.com/settings/keys"
    )

    # 4. torch import + version
    try:
        import torch
        ok = torch.__version__.startswith("2.10.0")
        all_ok &= check(f"torch {torch.__version__} (need 2.10.0+cu128)", ok)
        if torch.cuda.is_available():
            check(f"CUDA available (devices: {torch.cuda.device_count()})", True)
        else:
            check("CUDA not available", False, "Check NVIDIA drivers and CUDA install")
    except ImportError:
        all_ok &= check("torch not installed", False, "Run: uv sync")

    # 5. Private packages
    for pkg in ["dpg_evaluator", "geneval_evaluator", "t2v_metrics"]:
        try:
            __import__(pkg)
            all_ok &= check(f"{pkg} installed", True)
        except ImportError:
            all_ok &= check(f"{pkg} not installed", False, "Run: uv sync")

    print()
    if all_ok:
        print("All checks passed!")
    else:
        print("Some checks failed. See fixes above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
