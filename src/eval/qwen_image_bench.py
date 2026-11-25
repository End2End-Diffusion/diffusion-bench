"""Qwen-Image-Bench evaluator for T2I evaluation."""

import contextlib
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from qwen_image_bench_evaluator import QwenImageBench

# Level-1 dimensions -> reported metric suffix.
_DIM_TO_METRIC = {
    "Quality": "quality",
    "Aesthetics": "aesthetics",
    "Alignment": "alignment",
    "Real-world Fidelity": "real_world_fidelity",
    "Creative Generation": "creative_generation",
}


class QwenImageBenchEvaluator:
    """Lazy-loaded Qwen-Image-Bench judge.

    One judge pass yields the overall score plus five level-1 dimension scores.
    Dimensions are sparse (each prompt is only scored on the ones that apply), so
    missing scores are NaN and the caller masks them when averaging.
    """

    METRICS = ("total",) + tuple(_DIM_TO_METRIC.values())

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.evaluator = None

    def _ensure_loaded(self):
        if self.evaluator is None:
            # The judge loads onto the current CUDA device, so pin it to this rank's device.
            dev = torch.device(self.device)
            ctx = (
                torch.cuda.device(dev.index)
                if dev.type == "cuda" and dev.index is not None
                else contextlib.nullcontext()
            )
            # The judge calls torch.manual_seed() on load; fork the RNG so it doesn't
            # reset the sampling noise (and make it identical across ranks).
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with ctx, torch.random.fork_rng(devices=cuda_devices):
                self.evaluator = QwenImageBench()

    @torch.no_grad()
    def compute_batch_scores(self, images_np: np.ndarray, prompts: List[str]) -> Dict[str, torch.Tensor]:
        self._ensure_loaded()

        valid_indices, valid_images, valid_prompts = [], [], []
        for i, img in enumerate(images_np):
            if img.shape[0] > 0 and img.shape[1] > 0:
                valid_indices.append(i)
                valid_images.append(Image.fromarray(img))
                valid_prompts.append(prompts[i])
            else:
                print(f"[QwenImageBench] Warning: Skipping image {i} with invalid shape {img.shape}")

        scores = {m: torch.full((len(images_np),), float("nan")) for m in self.METRICS}
        if not valid_images:
            return scores

        # Results are aligned 1:1 with the input order.
        results = self.evaluator(valid_images, valid_prompts, show_progress=False)

        for idx, result in zip(valid_indices, results):
            if not result.get("valid", False):
                continue
            if result.get("total") is not None:
                scores["total"][idx] = float(result["total"])
            for dim_name, dim_data in result.get("dim_scores", {}).items():
                metric = _DIM_TO_METRIC.get(dim_name)
                if metric is not None and dim_data and dim_data.get("level1_score") is not None:
                    scores[metric][idx] = float(dim_data["level1_score"])
        return scores
