"""PRISM-Bench evaluator for T2I evaluation."""

from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from prism_bench_evaluator import EVAL_TYPES, TRACKS
from prism_bench_evaluator import PrismBenchEvaluator as _PrismJudge


class PrismBenchEvaluator:
    """Lazy-loaded PRISM-Bench judge (Qwen-VL).

    Each image is judged under two rubrics, ``alignment`` and ``aesthetic`` (0-10),
    in one pass. Returns both, their mean (``total``), and a per-track combined score.
    Per-track scores are sparse (each prompt belongs to one track), so off-track
    entries are NaN and the caller masks them when averaging.

    ``model_name`` is a JUDGE_MODELS short name (e.g. ``qwen2.5-vl-7b``) or an HF/local
    path. The package default (Qwen2.5-VL-72B) shards across all GPUs and does not fit
    per-rank online eval, so pick a judge that fits on one GPU.
    """

    METRICS = ("total", "alignment", "aesthetic") + tuple(TRACKS)

    def __init__(self, model_name: str = None, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.judge = None

    def _ensure_loaded(self):
        if self.judge is None:
            self.judge = _PrismJudge(model_name=self.model_name, device=self.device)

    @torch.no_grad()
    def compute_batch_scores(self, images_np: np.ndarray, prompts: List[str]) -> Dict[str, torch.Tensor]:
        self._ensure_loaded()

        valid_indices, valid_images, valid_prompts = [], [], []
        for i, img in enumerate(images_np):
            if img.shape[0] > 0 and img.shape[1] > 0:
                valid_indices.append(i)
                valid_images.append(Image.fromarray(img))
                # Don't strip: PRISM resolves prompts by exact match (some carry trailing spaces).
                valid_prompts.append(prompts[i])
            else:
                print(f"[PRISM] Warning: Skipping image {i} with invalid shape {img.shape}")

        # NaN = no score (invalid image, off-track, or unparseable judge output).
        scores = {m: torch.full((len(images_np),), float("nan")) for m in self.METRICS}
        if not valid_images:
            return scores

        # evaluate() returns len(images) * len(EVAL_TYPES) rows, image-major.
        results = self.judge.evaluate(valid_images, valid_prompts, show_progress=False)

        n_types = len(EVAL_TYPES)
        per_image = [{} for _ in valid_images]
        track_of = [None] * len(valid_images)
        for k, row in enumerate(results):
            vi = k // n_types
            track_of[vi] = row.get("track")
            if row.get("score") is not None:
                per_image[vi][row.get("eval_type")] = float(row["score"])

        for vi, orig in enumerate(valid_indices):
            a = per_image[vi].get("alignment")
            e = per_image[vi].get("aesthetic")
            if a is not None:
                scores["alignment"][orig] = a
            if e is not None:
                scores["aesthetic"][orig] = e
            present = [v for v in (a, e) if v is not None]
            if present:
                combined = float(np.mean(present))
                scores["total"][orig] = combined
                if track_of[vi] in scores:
                    scores[track_of[vi]][orig] = combined
        return scores
