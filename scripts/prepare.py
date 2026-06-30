#!/usr/bin/env python3
"""
Download and prepare datasets for nanogen training and evaluation.

Usage:
    uv run python scripts/prepare.py --data all
    uv run python scripts/prepare.py --data imagenet
    uv run python scripts/prepare.py --data t2i
    uv run python scripts/prepare.py --data eval
    uv run python scripts/prepare.py --data eval --data-dir /mnt/data
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

# repo_id -> local directory name
DATASETS = {
    "imagenet": ("diffusion-bench/imagenet", "imagenet"),
    "blip3o-256": ("diffusion-bench/blip3o-256", "blip3o-256"),
    "coco-30k": ("diffusion-bench/coco-30k", "mscoco"),
    "mjhq-30k": ("diffusion-bench/mjhq-30k", "mjhq"),
    "geneval": ("diffusion-bench/geneval", "geneval"),
    "dpgbench": ("diffusion-bench/dpgbench", "dpgbench"),
    "genaibench": ("diffusion-bench/genaibench", "genaibench"),
    "blip3o-original-lc": ("BLIP3o/BLIP3o-Pretrain-Long-Caption", "blip3o/long-caption"),
    "blip3o-original-sc": ("BLIP3o/BLIP3o-Pretrain-Short-Caption", "blip3o/short-caption"),
    "blip3o-original-jdb": ("BLIP3o/BLIP3o-Pretrain-JourneyDB", "blip3o/journeydb"),
}

GROUPS = {
    "imagenet": ["imagenet"],
    "t2i": ["blip3o-256"],
    "eval": ["coco-30k", "mjhq-30k", "geneval", "dpgbench", "genaibench"],
    "t2i-res512": ["blip3o-original-lc", "blip3o-original-sc", "blip3o-original-jdb"],
    "all": list(DATASETS.keys()),
}


def download_dataset(name: str, data_dir: Path) -> None:
    repo_id, local_name = DATASETS[name]
    local_dir = data_dir / local_name
    print(f"Downloading {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"Done: {name}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare nanogen datasets")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=list(GROUPS.keys()),
        help="Which dataset group to download",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root data directory (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    for name in GROUPS[args.data]:
        download_dataset(name, args.data_dir)


if __name__ == "__main__":
    main()
