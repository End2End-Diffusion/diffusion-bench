#!/usr/bin/env python3
"""
Download GPIC (stanford-vision-lab/gpic, ~100M permissive images) and center-crop it to
256/512px WebDataset shards readable by the `blip3o` loader ({key}.jpg / .txt caption / .json).
Raw shards (~13 TB) are streamed and deleted after cropping; output needs ~3.3 TB at 256px and
~10.4 TB at 512px. Reruns resume where they stopped.

FID files go to <data-dir>/gpic-<res>/eval/: the 50K val/test captions ({val,test}/) and Inception
reference stats ({val,test}_stats.npz). 256px uses GPIC's official stats; 512px stats are built
here over the same images with GPIC's recipe (center crop + bicubic resize). `--eval` prepares
only these. Note: GPIC's stats and its 50K captions are different subsets (~0.5 FID noise floor).

Usage:
    uv run python scripts/prepare_gpic.py [--res 256 512] [--eval]
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
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

REPO = "stanford-vision-lab/gpic"
SHARDS = {"val": 32, "test": 128, "train": 8000}
TB_TOTAL = {256: 3.3, 512: 10.4}  # all splits; measured ~33 / ~104 KB per image, ~101M images


def raw_samples(rel, tmp_root):
    """Download one raw shard and yield (key, json bytes, image bytes); the shard is deleted after."""
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        meta = None
        with tarfile.open(hf_hub_download(REPO, rel, repo_type="dataset", local_dir=tmp)) as tf:
            for m in tf:  # each {key}.json is followed by its {key}.jpg/.png
                key, ext = m.name.rsplit(".", 1)
                data = tf.extractfile(m).read()
                if ext == "json":
                    meta = data
                else:
                    yield key, meta, data


def add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def crop_shard(rel, data_dir, sizes):
    outs = [data_dir / f"gpic-{s}" / rel for s in sizes]
    parts = [o.with_name(o.name + ".partial") for o in outs]
    tars = [tarfile.open(p, "w") for p in parts]
    for key, meta, data in raw_samples(rel, data_dir / ".gpic_tmp"):
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        caption = json.loads(meta)["caption"].strip().encode()
        for s, tar in zip(sizes, tars):
            buf = io.BytesIO()
            ImageOps.fit(im, (s, s), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=95)
            add(tar, f"{key}.json", meta)
            add(tar, f"{key}.txt", caption)
            add(tar, f"{key}.jpg", buf.getvalue())
    for tar, p, o in zip(tars, parts, outs):
        tar.close()
        p.replace(o)


def prepare_shards(data_dir, sizes, workers):
    todo = [f"{sp}/gpic_{sp}_{i:05d}.tar" for sp, n in SHARDS.items() for i in range(n)]
    todo = [r for r in todo if not all((data_dir / f"gpic-{s}" / r).exists() for s in sizes)]
    need = sum(TB_TOTAL[s] for s in sizes) * len(todo) / sum(SHARDS.values())
    free = shutil.disk_usage(data_dir).free / 1e12
    print(f"WARNING: GPIC at {sizes}px: {len(todo)} shards to go, needs ~{need:.1f} TB "
          f"on {data_dir.resolve()} ({free:.1f} TB free).", flush=True)
    if need > free:
        sys.exit("Not enough disk space; pass --data-dir on a larger disk.")
    for s in sizes:
        for sp in SHARDS:
            (data_dir / f"gpic-{s}" / sp).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(data_dir / ".gpic_tmp", ignore_errors=True)
    (data_dir / ".gpic_tmp").mkdir()
    failed = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(crop_shard, r, data_dir, sizes): r for r in todo}
        for i, f in enumerate(as_completed(futures), 1):
            if f.exception():
                failed.append(futures[f])
                print(f"FAILED {futures[f]}: {f.exception()}", flush=True)
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} shards done, {len(failed)} failed", flush=True)
    shutil.rmtree(data_dir / ".gpic_tmp", ignore_errors=True)
    if failed:
        sys.exit(f"{len(failed)} shards failed; rerun to resume.")


def ref_crops(rel, keys, size, tmp_root):  # GPIC reference recipe: square center crop + bicubic resize
    out = []
    for key, _, data in raw_samples(rel, tmp_root):
        if key in keys:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            w, h = im.size
            c, l, t = min(w, h), (w - min(w, h)) // 2, (h - min(w, h)) // 2
            out.append(np.asarray(im.crop((l, t, l + c, t + c)).resize((size, size), Image.Resampling.BICUBIC)))
    return out


def ref_stats(data_dir, split, keys, size, workers):
    """Inception moments of the protocol images, with the same extractor as src/eval/fid.py."""
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    tmp_root = data_dir / ".gpic_tmp_eval"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True)
    feats = []
    with ProcessPoolExecutor(workers) as pool:
        rels = [f"{split}/gpic_{split}_{i:05d}.tar" for i in range(SHARDS[split])]
        for f in as_completed([pool.submit(ref_crops, r, keys, size, tmp_root) for r in rels]):
            crops = f.result()
            for i in range(0, len(crops), 100):
                x = torch.from_numpy(np.stack(crops[i:i + 100])).permute(0, 3, 1, 2).to(device)
                with torch.no_grad():
                    feats.append(fe(x)[0].double().cpu().numpy())
    shutil.rmtree(tmp_root, ignore_errors=True)
    feats = np.concatenate(feats)
    print(f"GPIC {split} {size}px reference: {len(feats)}/{len(keys)} protocol images", flush=True)
    return dict(inception_mu=feats.mean(0), inception_sigma=np.cov(feats, rowvar=False))


def prepare_eval(data_dir, sizes, workers):
    from datasets import Dataset

    for split in ("val", "test"):
        rows = [json.loads(line) for line in open(hf_hub_download(REPO, f"gpic_{split}_50k.jsonl", repo_type="dataset"))]
        official = None  # GPIC's 256px stats; note they cover a different 50K subset than the captions
        for size in sizes:
            out = data_dir / f"gpic-{size}" / "eval"
            stats = out / f"{split}_stats.npz"
            if not (out / split).exists():
                Dataset.from_dict({"text": [r["caption"].strip() for r in rows]}).save_to_disk(str(out / split))
            if stats.exists():
                continue
            official = official or hf_hub_download(REPO, f"reference_stats/{split}_stats.npz", repo_type="dataset",
                                                   local_dir=data_dir / ".gpic_ref")
            if size == 256:
                shutil.copy(official, stats)
            else:  # same images as the official stats, at this resolution
                np.savez(stats, **ref_stats(data_dir, split, set(np.load(official)["key"].tolist()), size, workers))
            print(f"GPIC eval {split} {size}px -> {out}", flush=True)
    shutil.rmtree(data_dir / ".gpic_ref", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Download and center-crop GPIC")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent.parent / "data")
    parser.add_argument("--res", type=int, nargs="+", default=[256, 512], choices=[256, 512])
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() // 2))
    parser.add_argument("--eval", action="store_true", help="Only prepare the FID captions + reference stats")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    prepare_eval(args.data_dir, args.res, args.workers)
    if not args.eval:
        prepare_shards(args.data_dir, args.res, args.workers)


if __name__ == "__main__":
    main()
