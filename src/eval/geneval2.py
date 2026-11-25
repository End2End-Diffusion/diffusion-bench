"""GenEval2 (Soft-TIFA) evaluator for T2I evaluation."""

from typing import Dict, List

import numpy as np
import torch
from geneval2_evaluator import evaluate_pairs, fetch_metadata, load_models
from PIL import Image


class GenEval2Evaluator:
    """Lazy-loaded GenEval2 Soft-TIFA judge (Qwen3-VL).

    One judge pass yields all four GenEval2 metrics per image (Soft-TIFA AM/GM,
    hard TIFA, whole-prompt VQAScore), so they are returned together.
    """

    METRICS = ("soft_tifa_am", "soft_tifa_gm", "tifa", "vqascore")

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = None
        self.metadata = None

    def _ensure_loaded(self):
        if self.models is None:
            self.models = load_models(device=self.device)
        if self.metadata is None:
            # fetch_metadata() returns a list of prompt dicts; index by prompt text.
            self.metadata = {m["prompt"]: m for m in fetch_metadata()}

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
                print(f"[GenEval2] Warning: Skipping image {i} with invalid shape {img.shape}")

        scores = {m: torch.zeros(len(images_np)) for m in self.METRICS}
        if not valid_images:
            return scores

        metadatas = [self.metadata[p] for p in valid_prompts]
        results = evaluate_pairs(images=valid_images, metadata_list=metadatas, models=self.models, show_progress=False)

        # evaluate_pairs is positional; make sure no image was judged against another prompt.
        for r, p in zip(results, valid_prompts):
            assert r["prompt"] == p, f"GenEval2 alignment error: judged {r['prompt']!r} != expected {p!r}"

        for idx, result in zip(valid_indices, results):
            for m in self.METRICS:
                scores[m][idx] = float(result[m])
        return scores
