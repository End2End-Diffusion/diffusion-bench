#!/usr/bin/env python3
"""
Investigate a HuggingFace VAE before integrating it.

Reports: diffusers class, config attributes, tensor format,
normalization parameters, latent shape, and encode-decode round trip.

Usage:
    uv run python skills/stage1-add-vae/scripts/investigate_vae.py --model Qwen/Qwen-Image --subfolder vae
    uv run python skills/stage1-add-vae/scripts/investigate_vae.py --model stabilityai/stable-diffusion-3.5-large --subfolder vae
    uv run python skills/stage1-add-vae/scripts/investigate_vae.py --model black-forest-labs/FLUX.1-dev --subfolder vae
"""

import argparse
import sys


CLASSES_TO_TRY = [
    "AutoencoderKL",
    "AutoencoderKLQwenImage",
    "AutoencoderKLWan",
    "AutoencoderKLFlux2",
]


def try_load(model_path: str, subfolder: str | None):
    """Try loading with AutoencoderKL first, then known specialized classes."""
    load_kwargs = {}
    if subfolder:
        load_kwargs["subfolder"] = subfolder

    for cls_name in CLASSES_TO_TRY:
        try:
            mod = __import__("diffusers", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError):
            continue

        try:
            vae = cls.from_pretrained(model_path, **load_kwargs)
            return vae, cls_name
        except Exception as e:
            err_msg = str(e)
            if "expected shape" in err_msg or "size mismatch" in err_msg:
                print(f"  [{cls_name}] shape mismatch — skipping")
            else:
                print(f"  [{cls_name}] failed: {err_msg[:120]}")
            continue

    return None, None


def investigate(model_path: str, subfolder: str | None, resolution: int):
    import torch

    print(f"Investigating: {model_path}" + (f" (subfolder={subfolder})" if subfolder else ""))
    print("=" * 60)

    # 1. Load
    print("\n1. Loading VAE...")
    vae, cls_name = try_load(model_path, subfolder)
    if vae is None:
        print("   FAILED: Could not load with any known class.")
        print(f"   Tried: {', '.join(CLASSES_TO_TRY)}")
        return 1

    print(f"   Loaded with: {cls_name}")
    needs_custom_class = cls_name != "AutoencoderKL"

    # 2. Config
    print("\n2. Config attributes:")
    config = vae.config
    for key in ["latents_mean", "latents_std", "scaling_factor", "shift_factor",
                "latent_channels", "in_channels", "out_channels", "z_dim", "base_dim"]:
        val = getattr(config, key, None)
        if val is not None:
            if isinstance(val, list) and len(val) > 6:
                print(f"   {key}: [{val[0]}, {val[1]}, ... {val[-1]}] (len={len(val)})")
            else:
                print(f"   {key}: {val}")

    # 3. Normalization
    print("\n3. Normalization:")
    has_latents_mean = hasattr(config, "latents_mean") and config.latents_mean is not None
    has_latents_std = hasattr(config, "latents_std") and config.latents_std is not None
    has_shift = hasattr(config, "shift_factor") and config.shift_factor is not None
    has_scale = hasattr(config, "scaling_factor") and config.scaling_factor is not None
    has_bn = hasattr(vae, "bn")

    if has_latents_mean and has_latents_std:
        print("   Per-channel (latents_mean/latents_std) — auto-handled by base VAE class")
    elif has_shift or has_scale:
        print(f"   Scalar (shift={getattr(config, 'shift_factor', None)}, "
              f"scale={getattr(config, 'scaling_factor', None)}) — auto-handled by base VAE class")
    else:
        print("   None found in config — set scaling_factor/shift_factor in VAEConfig manually")
    if has_bn:
        print("   WARNING: has BatchNorm (vae.bn) — needs fully custom encode/decode like Flux2VAE")

    # 4. Tensor format + round trip
    print(f"\n4. Tensor format probe (resolution={resolution}):")
    vae = vae.eval()
    x_4d = torch.randn(1, 3, resolution, resolution)
    x_5d = torch.randn(1, 3, 1, resolution, resolution)

    works_4d = False
    works_5d = False
    latent_channels = None
    downsample = None

    with torch.no_grad():
        for label, x in [("4D", x_4d), ("5D", x_5d)]:
            try:
                enc_out = vae.encode(x)
                z = enc_out.latent_dist.mode()
                dec_out = vae.decode(z)
                x_rec = dec_out.sample
                z_shape = list(z.shape)
                print(f"   {label} input {list(x.shape)} -> latent {z_shape} -> recon {list(x_rec.shape)} — WORKS")
                if label == "4D":
                    works_4d = True
                else:
                    works_5d = True
                if latent_channels is None:
                    latent_channels = z_shape[1]
                    latent_h = z_shape[-1]
                    downsample = resolution // latent_h
            except Exception as e:
                print(f"   {label} input {list(x.shape)} -> FAILED: {str(e)[:100]}")

    needs_unsqueeze = (not works_4d) and works_5d

    # 5. Summary
    short_name = model_path.split("/")[-1].lower().replace("_", "-")
    print(f"\n5. VAEConfig entry:")
    print(f'    "{short_name}": VAEConfig(')
    print(f'        pretrained_path="{model_path}",')
    if subfolder:
        print(f'        subfolder="{subfolder}",')
    if latent_channels:
        print(f"        latent_channels={latent_channels},")
    if downsample:
        print(f"        downsample_factor={downsample},")
    if not has_latents_mean and has_scale:
        print(f"        scaling_factor={config.scaling_factor},")
    if not has_latents_mean and has_shift:
        print(f"        shift_factor={config.shift_factor},")
    print("    ),")

    # 6. Integration notes
    print("\n6. What you need:")
    if not needs_custom_class and works_4d and not has_bn:
        print("   SIMPLE: Just add the VAEConfig entry above. Use base VAE class.")
        print("   No custom class needed.")
    else:
        print("   CUSTOM CLASS NEEDED. Reasons:")
        if needs_custom_class:
            print(f"   - Uses {cls_name} (not AutoencoderKL) -> override _load_vae()")
        if needs_unsqueeze:
            print("   - Needs 5D tensors -> override _vae_encode() and _vae_decode() with unsqueeze/squeeze")
        if has_bn:
            print("   - Has BatchNorm -> fully override encode()/decode() (see Flux2VAE)")
        print("   See: skills/stage1-add-vae/references/custom-classes.md")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Investigate a HuggingFace VAE")
    parser.add_argument("--model", required=True, help="HuggingFace model path (e.g. Qwen/Qwen-Image)")
    parser.add_argument("--subfolder", default=None, help="Subfolder within model (e.g. vae)")
    parser.add_argument("--resolution", type=int, default=256, help="Test resolution")
    args = parser.parse_args()

    sys.exit(investigate(args.model, args.subfolder, args.resolution))


if __name__ == "__main__":
    main()
