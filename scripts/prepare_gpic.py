#!/usr/bin/env python3
"""
Download GPIC (stanford-vision-lab/gpic, ~100M permissive images) and center-crop it to
256/512px WebDataset shards readable by the `blip3o` loader ({key}.jpg / .txt caption / .json).
Raw shards (~13 TB) are streamed and deleted after cropping; output needs ~3.3 TB at 256px and
~10.4 TB at 512px. Reruns resume where they stopped.

Eval sets go to <data-dir>/gpic-<res>/eval/{val,test}/: GPIC's 50K protocol captions with their
images (GPIC's recipe: square center crop + bicubic resize, stored as PNG), so FID references are
computed from the same images. `--eval` prepares only these.

Usage:
    uv run python scripts/prepare_gpic.py [--res 256 512] [--eval]
"""

import argparse
import io
import json
import os
import shutil
import signal
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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


def run_pool(fn, tasks, workers, on_done):
    """Run fn(*task) in worker processes, calling on_done(i, task, future) as each finishes.
    Ctrl-C cancels queued tasks, lets running ones finish, then exits (rerun to resume)."""
    pool = ProcessPoolExecutor(workers, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN))
    futures = {pool.submit(fn, *t): t for t in tasks}
    try:
        for i, f in enumerate(as_completed(futures), 1):
            on_done(i, futures[f], f)
    except KeyboardInterrupt:
        print("Interrupted: finishing running tasks, then exiting; rerun to resume.", flush=True)
        pool.shutdown(cancel_futures=True)
        sys.exit(130)
    pool.shutdown()


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

    def done(i, task, f):
        if f.exception():
            failed.append(task[0])
            print(f"FAILED {task[0]}: {f.exception()}", flush=True)
        if i % 20 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} shards done, {len(failed)} failed", flush=True)

    run_pool(crop_shard, [(r, data_dir, sizes) for r in todo], workers, done)
    shutil.rmtree(data_dir / ".gpic_tmp", ignore_errors=True)
    if failed:
        sys.exit(f"{len(failed)} shards failed; rerun to resume.")


def eval_crops(rel, keys, sizes, tmp_root):  # GPIC reference recipe: square center crop + bicubic resize
    out = {}
    for key, _, data in raw_samples(rel, tmp_root):
        if key in keys:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            w, h = im.size
            c, l, t = min(w, h), (w - min(w, h)) // 2, (h - min(w, h)) // 2
            square, out[key] = im.crop((l, t, l + c, t + c)), []
            for s in sizes:
                buf = io.BytesIO()
                square.resize((s, s), Image.Resampling.BICUBIC).save(buf, "PNG")
                out[key].append(buf.getvalue())
    return out


def prepare_eval(data_dir, sizes, workers):
    from datasets import Dataset, Features, Image as HFImage, Value

    tmp_root = data_dir / ".gpic_tmp_eval"
    for split in ("val", "test"):
        outs = [data_dir / f"gpic-{s}" / "eval" / split for s in sizes]
        if all(o.exists() for o in outs):
            continue
        rows = [json.loads(line) for line in open(hf_hub_download(REPO, f"gpic_{split}_50k.jsonl", repo_type="dataset"))]
        keys, images = {r["key"] for r in rows}, {}
        shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir()
        rels = [f"{split}/gpic_{split}_{i:05d}.tar" for i in range(SHARDS[split])]
        run_pool(eval_crops, [(r, keys, sizes, tmp_root) for r in rels], workers,
                 lambda i, task, f: images.update(f.result()))
        rows = [r for r in rows if r["key"] in images]  # a few protocol images were removed upstream
        features = Features({"image": HFImage(), "text": Value("string")})
        for i, out in enumerate(outs):
            partial = out.with_name(out.name + ".partial")  # renamed when complete, so reruns can resume
            shutil.rmtree(partial, ignore_errors=True)
            Dataset.from_dict({"image": [{"bytes": images[r["key"]][i], "path": None} for r in rows],
                               "text": [r["caption"].strip() for r in rows]}, features=features).save_to_disk(str(partial))
            shutil.rmtree(out, ignore_errors=True)
            partial.rename(out)
            print(f"GPIC eval {split}: {len(rows)} images + captions -> {out}", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Download and center-crop GPIC")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent.parent / "data")
    parser.add_argument("--res", type=int, nargs="+", default=[256, 512], choices=[256, 512])
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() // 2))
    parser.add_argument("--eval", action="store_true", help="Only prepare the eval sets")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    prepare_eval(args.data_dir, args.res, args.workers)
    if not args.eval:
        prepare_shards(args.data_dir, args.res, args.workers)


if __name__ == "__main__":
    main()
