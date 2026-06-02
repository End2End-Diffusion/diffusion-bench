"""Eval module — re-exports from submodules."""

from .ref_iqa import calculate_psnr, calculate_lpips, calculate_ssim
from .fid import calculate_rfid, calculate_gfid
from .clipscore import CLIPScoreEvaluator
from .vqascore import VQAScoreEvaluator
from .geneval import GenEvalEvaluator
from .dpgbench import DPGEvaluator

from .reconstruction import compute_reconstruction_metrics, evaluate_reconstruction_distributed
from .generation import evaluate_generation_distributed

import numpy as np
import torch
from typing import Dict


def compute_generation_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, float]:
    device_str = "cuda" if device.type == "cuda" else "cpu"
    fid = calculate_gfid(rec_arr, ref_arr, batch_size, device_str)
    return {'fid': fid}


__all__ = [
    # ref_iqa
    "calculate_psnr", "calculate_lpips", "calculate_ssim",
    # fid
    "calculate_rfid", "calculate_gfid",
    # evaluators
    "CLIPScoreEvaluator", "VQAScoreEvaluator", "GenEvalEvaluator", "DPGEvaluator",
    # reconstruction
    "compute_reconstruction_metrics", "evaluate_reconstruction_distributed",
    # generation
    "evaluate_generation_distributed", "compute_generation_metrics",
]
