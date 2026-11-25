#!/usr/bin/env python3
"""
Download GPIC (stanford-vision-lab/gpic, ~100M permissive images) and center-crop it to
256/512px WebDataset shards readable by the `blip3o` loader ({key}.jpg / .txt caption / .json).

Raw shards (~13 TB for train) are streamed one by one and deleted after cropping; the
cropped output needs ~3.3 TB at 256px and ~10.4 TB at 512px. Reruns resume where they stopped.

Usage:
    uv run python scripts/prepare_gpic.py                        # all splits, 256 + 512
    uv run python scripts/prepare_gpic.py --res 256 --splits val test
"""

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

GPIC_REPO = "stanford-vision-lab/gpic"
GPIC_SHARDS = {"val": 32, "test": 128, "train": 8000}
GPIC_IMAGES = {"val": 200_000, "test": 1_000_000, "train": 100_000_000}
GPIC_KB_PER_IMAGE = {256: 33, 512: 104}  # measured on a val shard
GPIC_RAW_SHARD_GB = 1.6


def _add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def gpic_shard(rel, data_dir, sizes):
    outs = [data_dir / f"gpic-{s}" / rel for s in sizes]
    if all(o.exists() for o in outs):
        return 0, 0
    kept = skipped = 0
    with tempfile.TemporaryDirectory(dir=data_dir / ".gpic_tmp") as tmp, ExitStack() as stack:
        src = hf_hub_download(GPIC_REPO, rel, repo_type="dataset", local_dir=tmp)
        parts = [o.with_name(o.name + ".partial") for o in outs]
        tars = [stack.enter_context(tarfile.open(p, "w")) for p in parts]
        meta = None
        with tarfile.open(src) as tf:
            for m in tf:  # each {key}.json is followed by its {key}.jpg/.png
                key, ext = m.name.rsplit(".", 1)
                data = tf.extractfile(m).read()
                if ext == "json":
                    meta = (key, data)
                    continue
                try:
                    assert meta and meta[0] == key, f"image without metadata: {m.name}"
                    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                    crops = []
                    for s in sizes:
                        buf = io.BytesIO()
                        ImageOps.fit(im, (s, s), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=95)
                        crops.append(buf.getvalue())
                except Exception as e:
                    print(f"skip {rel}:{m.name}: {e}", flush=True)
                    skipped += 1
                    continue
                caption = json.loads(meta[1])["caption"].strip().encode()
                for tar, crop in zip(tars, crops):
                    _add(tar, f"{key}.json", meta[1])
                    _add(tar, f"{key}.txt", caption)
                    _add(tar, f"{key}.jpg", crop)
                kept += 1
        stack.close()
        for p, o in zip(parts, outs):
            p.replace(o)
    return kept, skipped


def prepare_gpic(data_dir, sizes, splits, workers):
    shards = [f"{sp}/gpic_{sp}_{i:05d}.tar" for sp in splits for i in range(GPIC_SHARDS[sp])]
    todo = [r for r in shards if not all((data_dir / f"gpic-{s}" / r).exists() for s in sizes)]
    images = sum(GPIC_IMAGES[sp] for sp in splits) * len(todo) / max(len(shards), 1)
    need = images * sum(GPIC_KB_PER_IMAGE[s] for s in sizes) / 1e9 + workers * GPIC_RAW_SHARD_GB / 1e3
    data_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_dir).free / 1e12
    print(f"WARNING: GPIC {splits} at {sizes}px: {len(todo)}/{len(shards)} shards to go, "
          f"~{images / 1e6:.1f}M images, needs ~{need:.1f} TB on {data_dir.resolve()} ({free:.1f} TB free).\n"
          f"Raw shards (~13 TB for train) are streamed and deleted; set --workers to trade CPU for speed.")
    if need > free:
        sys.exit("Not enough disk space for GPIC; free space or pass --data-dir on a larger disk.")
    for s in sizes:
        for sp in splits:
            (data_dir / f"gpic-{s}" / sp).mkdir(parents=True, exist_ok=True)
            for stale in (data_dir / f"gpic-{s}" / sp).glob("*.partial"):
                stale.unlink()
    shutil.rmtree(data_dir / ".gpic_tmp", ignore_errors=True)
    (data_dir / ".gpic_tmp").mkdir()
    kept = skipped = 0
    failed = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(gpic_shard, r, data_dir, sizes): r for r in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                k, sk = f.result()
                kept, skipped = kept + k, skipped + sk
            except Exception as e:
                failed.append(futures[f])
                print(f"FAILED {futures[f]}: {e}", flush=True)
            if i == 1 or i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} shards, {kept} images kept, {skipped} skipped, {len(failed)} failed", flush=True)
    shutil.rmtree(data_dir / ".gpic_tmp", ignore_errors=True)
    if failed:
        sys.exit(f"{len(failed)} shards failed; rerun the same command to resume.")


def main():
    parser = argparse.ArgumentParser(description="Download and center-crop GPIC")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Outputs go to <data-dir>/gpic-<res>/<split>/")
    parser.add_argument("--res", type=int, nargs="+", default=[256, 512], choices=[256, 512])
    parser.add_argument("--splits", nargs="+", default=["val", "test", "train"], choices=list(GPIC_SHARDS))
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() // 2))
    args = parser.parse_args()
    prepare_gpic(args.data_dir, args.res, args.splits, args.workers)


if __name__ == "__main__":
    main()
