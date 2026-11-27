"""MFU as in PixelCraft: denoiser fwd+bwd only (no text encoder / VAE).
MFU = samples/s * FLOPs per sample / (num GPUs * peak); the forward is counted once with FlopCounterMode, x3 for fwd+bwd."""
import os

import torch
from torch.utils.flop_counter import FlopCounterMode

_PEAK_BF16 = {"B300": 2.25e15, "B200": 2.25e15, "H200": 989e12, "H100": 989e12, "L40S": 362e12, "A100": 312e12}  # dense
FLOPS_PER_SAMPLE = None


def peak_flops():  # per GPU; override with DIFFUSIONBENCH_PEAK_TFLOPS
    if os.environ.get("DIFFUSIONBENCH_PEAK_TFLOPS"):
        return float(os.environ["DIFFUSIONBENCH_PEAK_TFLOPS"]) * 1e12
    name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
    return next((v for k, v in _PEAK_BF16.items() if k in name), None)


def measure(model, z, model_kwargs, n=2):
    global FLOPS_PER_SAMPLE
    t, kw = torch.rand(n, device=z.device), {k: v[:n] for k, v in model_kwargs.items() if v is not None}
    if getattr(model, "is_meanflow", False):
        kw["t_abs"] = t
    with FlopCounterMode(display=False) as c, torch.no_grad():
        model(z[:n], t, **kw)
    FLOPS_PER_SAMPLE = 3 * c.get_total_flops() / n
