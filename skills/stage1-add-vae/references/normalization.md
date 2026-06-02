# VAE Latent Normalization

The base `VAE.__init__()` automatically resolves shift and scale factors from the loaded VAE config. Understanding this is important when debugging reconstruction quality (e.g. bad rFID with good PSNR/SSIM).

## Resolution order

### Shift factor (applied as: `z = (z - shift) * scale`)

1. `vae.config.latents_mean` — per-channel tensor, reshaped to `(1, C, 1, 1)`. Used by newer VAEs (Qwen-Image, SD3.5, FLUX).
2. `VAEConfig.shift_factor` — scalar override in `VAE_CONFIGS` dict. Used for older VAEs (sdvae-ema/mse).
3. `vae.config.shift_factor` — scalar from model config. Fallback default: `0.0`.

### Scale factor (applied as: `z = (z - shift) * scale`)

1. `1 / vae.config.latents_std` — per-channel tensor, reshaped to `(1, C, 1, 1)`. Inverse of std.
2. `VAEConfig.scaling_factor` — scalar override in `VAE_CONFIGS` dict.
3. `vae.config.scaling_factor` — scalar from model config. Fallback default: `1.0`.

## Decode (inverse)

```python
z = z / scaling_factor + shift_factor
```

## Debugging normalization issues

If a VAE has good PSNR/SSIM but bad rFID:

1. Check what `vae.config` actually exposes:
   ```python
   from diffusers import AutoencoderKL  # or specialized class
   vae = AutoencoderKL.from_pretrained("org/model", subfolder="vae")
   print(vae.config)
   ```

2. Verify the shift/scale values loaded correctly:
   ```python
   model = QwenVAE(vae_type="qwen-vae")
   print("shift:", model.shift_factor)
   print("scale:", model.scaling_factor)
   ```

3. If the config has `latents_mean`/`latents_std`, the base class handles it automatically — no need to set `scaling_factor`/`shift_factor` in `VAEConfig`.

4. If the config does NOT have these fields and uses a non-standard normalization scheme, you may need to:
   - Set explicit values in `VAEConfig`
   - Or override `encode()`/`decode()` entirely (like Flux2VAE does with BatchNorm)

## Example: Qwen-Image VAE config

```
latents_mean: [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, ...]
latents_std:  [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, ...]
```

These are 16-element vectors (one per latent channel). The base class picks them up automatically via the first branch of the resolution order.
