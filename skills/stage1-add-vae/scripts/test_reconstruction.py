#!/usr/bin/env python3
"""
Visual reconstruction test for VAEs.

Loads VAE(s), reconstructs ImageNet val images, prints per-image PSNR/SSIM,
and saves side-by-side comparison images (original | recon_1 | recon_2 | ...).

Uses the same config-driven instantiation as the eval pipeline so any
registered VAE type works without manual class mapping.

Usage (requires PYTHONPATH=src):
    PYTHONPATH=src:$PYTHONPATH uv run python skills/stage1-add-vae/scripts/test_reconstruction.py --vae-type qwen-vae
    PYTHONPATH=src:$PYTHONPATH uv run python skills/stage1-add-vae/scripts/test_reconstruction.py --vae-type qwen-vae e2e-qwen-vae flux
    PYTHONPATH=src:$PYTHONPATH uv run python skills/stage1-add-vae/scripts/test_reconstruction.py --vae-type qwen-vae --num-images 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data import ImageNetHFDataset
from stage1.vae import VAE_CONFIGS


def build_vae(vae_type: str, resolution: int):
    """Build a VAE from the registry, auto-detecting the correct class."""
    if vae_type not in VAE_CONFIGS:
        raise ValueError(f"Unknown vae_type '{vae_type}'. Available: {sorted(VAE_CONFIGS.keys())}")

    # Try loading via eval config first (most reliable — matches eval pipeline exactly)
    config_files = list(Path("experiments").rglob(f"stage1-eval/configs/{vae_type}.yaml"))
    if config_files:
        from omegaconf import OmegaConf
        from configs.stage1 import Stage1Config
        from utils.model_utils import instantiate_from_config
        config: Stage1Config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage1Config), OmegaConf.load(config_files[0])))
        return instantiate_from_config(config.stage_1)

    # Fallback: try base VAE class, which works for standard AutoencoderKL types
    from stage1 import VAE
    return VAE(vae_type=vae_type, resolution=resolution)


def load_images(data_dir: str, num_images: int, resolution: int):
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    dataset = ImageNetHFDataset(data_dir=data_dir, split="val", transform=transform, condition_type="label")
    loader = DataLoader(dataset, batch_size=num_images, shuffle=False, num_workers=4)
    images, labels = next(iter(loader))
    names = [f"img_{i:04d}_label_{labels[i].item()}" for i in range(len(labels))]
    return images, names


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM using torchmetrics (matches eval pipeline)."""
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure
    # Convert HWC uint8 [0,255] -> BCHW float [0,1]
    t_a = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t_b = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return structural_similarity_index_measure(preds=t_b, target=t_a, data_range=1.0).item()


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return t.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def save_side_by_side(orig: torch.Tensor, recons: dict, name: str, out_dir: Path,
                      display_size: int, metrics: dict):
    vae_names = list(recons.keys())
    ncols = 1 + len(vae_names)
    pad, header_h = 8, 50
    w = ncols * display_size + (ncols + 1) * pad
    h = display_size + 2 * pad + header_h

    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Labels
    labels = ["Original"] + [
        f"{v}\nPSNR={metrics[v]['psnr']:.1f} SSIM={metrics[v]['ssim']:.3f}" for v in vae_names
    ]
    for col, label in enumerate(labels):
        x = pad + col * (display_size + pad) + display_size // 2
        for i, line in enumerate(label.split("\n")):
            draw.text((x, 5 + i * 16), line, fill=(0, 0, 0), anchor="mt")

    # Images
    y = header_h + pad
    orig_pil = Image.fromarray(to_uint8(orig)).resize((display_size, display_size), Image.LANCZOS)
    canvas.paste(orig_pil, (pad, y))
    for col, v in enumerate(vae_names):
        x = pad + (col + 1) * (display_size + pad)
        rec_pil = Image.fromarray(to_uint8(recons[v])).resize((display_size, display_size), Image.LANCZOS)
        canvas.paste(rec_pil, (x, y))

    canvas.save(out_dir / f"{name}.png")


def main():
    parser = argparse.ArgumentParser(description="Visual VAE reconstruction test")
    parser.add_argument("--vae-type", nargs="+", required=True)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data/imagenet")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--display-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]

    print(f"Loading {args.num_images} images from {args.data_dir}...")
    images, names = load_images(args.data_dir, args.num_images, args.resolution)
    images = images.to(device=device, dtype=dtype)

    recons = {}
    for vt in args.vae_type:
        print(f"Loading {vt}...")
        model = build_vae(vt, args.resolution).to(device).eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype):
            recons[vt] = model(images).clamp(0, 1).float().cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print metrics table
    out_dir = Path(args.output_dir or f"skills/stage1-add-vae/outputs/recon_{'_vs_'.join(args.vae_type)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    header = f"{'Image':<30}" + "".join(f"{v:<25}" for v in args.vae_type)
    print(f"\n{header}\n{'-' * len(header)}")

    images_cpu = images.float().cpu()
    for i, name in enumerate(names):
        orig_np = to_uint8(images_cpu[i])
        img_recons = {}
        img_metrics = {}
        row = f"{name:<30}"
        for vt in args.vae_type:
            rec_np = to_uint8(recons[vt][i])
            p, s = psnr(orig_np, rec_np), ssim(orig_np, rec_np)
            img_recons[vt] = recons[vt][i]
            img_metrics[vt] = {"psnr": p, "ssim": s}
            row += f"PSNR={p:5.1f} SSIM={s:.3f}    "
        print(row)
        save_side_by_side(images_cpu[i], img_recons, name, out_dir,
                          display_size=args.display_size, metrics=img_metrics)

    print(f"\nSaved {len(names)} comparisons to {out_dir}/")

    # Clean up output directory
    import shutil
    shutil.rmtree(out_dir)


if __name__ == "__main__":
    main()
