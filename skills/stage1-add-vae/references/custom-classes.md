# Custom VAE Classes

When a VAE doesn't work with standard `AutoencoderKL` or 4D tensors, create a subclass of `VAE` in `src/stage1/vae.py`.

## Hook methods available

The base `VAE` class provides three override points so subclasses don't need to duplicate normalization or preprocessing logic:

| Method | Default behavior | Override when |
|---|---|---|
| `_load_vae()` | `AutoencoderKL.from_pretrained(...)` | Different diffusers class needed |
| `_vae_encode(x)` | `self.vae.encode(x).latent_dist` | Custom tensor format (e.g. 5D) |
| `_vae_decode(z)` | `self.vae.decode(z).sample` | Custom tensor format (e.g. 5D) |

The base `encode()` and `decode()` handle preprocessing ([0,1]→[-1,1]), normalization (shift/scale), and sampling (mode/sample) — subclasses only override the raw VAE interaction.

## Example 1: QwenVAE (different class + 5D tensors)

**Problem:** Qwen-Image VAE uses `AutoencoderKLQwenImage` (not `AutoencoderKL`) and expects 5D tensors `(B, C, T, H, W)` with a temporal dimension.

**Solution:** Override `_load_vae` for the correct class, override `_vae_encode`/`_vae_decode` to unsqueeze/squeeze the temporal dimension.

```python
class QwenVAE(VAE):
    def __init__(self, vae_type="qwen-vae", resolution=256, eps=1e-5, sample_mode="mode"):
        super().__init__(vae_type=vae_type, resolution=resolution, eps=eps, sample_mode=sample_mode)

    def _load_vae(self):
        from diffusers import AutoencoderKLQwenImage
        load_kwargs = {"subfolder": self._subfolder}
        self.vae = AutoencoderKLQwenImage.from_pretrained(self._pretrained_path, **load_kwargs).eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def _vae_encode(self, x):
        x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        posterior = self.vae.encode(x).latent_dist
        posterior.mean = posterior.mean.squeeze(2)
        posterior.logvar = posterior.logvar.squeeze(2)
        return posterior

    def _vae_decode(self, z):
        z = z.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        x = self.vae.decode(z).sample
        return x.squeeze(2)  # (B, 3, 1, H, W) -> (B, 3, H, W)
```

**Key detail:** When squeezing the posterior, update both `mean` and `logvar` so that `mode()` and `sample()` return 4D tensors.

**Supports multiple config entries:** Both `"qwen-vae"` and `"e2e-qwen-vae"` use the same `QwenVAE` class — the `vae_type` parameter selects the config.

## Example 2: Flux2VAE (different class + patchify + BatchNorm)

**Problem:** Flux2 uses `AutoencoderKLFlux2`, outputs 32-channel latents at 32x32, then patchifies to 128 channels at 16x16, and normalizes with BatchNorm instead of shift/scale.

**Solution:** Override `_load_vae` for the correct class, fully override `encode()`/`decode()` because normalization is completely different (BatchNorm replaces shift/scale).

```python
class Flux2VAE(VAE):
    def __init__(self, resolution=256, eps=1e-5, sample_mode="mode"):
        super().__init__(vae_type="flux2", resolution=resolution, eps=eps, sample_mode=sample_mode)
        self.register_buffer('bn_mean', self.vae.bn.running_mean.clone())
        self.register_buffer('bn_std', torch.sqrt(self.vae.bn.running_var.clone() + eps))

    def _load_vae(self):
        from diffusers import AutoencoderKLFlux2
        self.vae = AutoencoderKLFlux2.from_pretrained(self._pretrained_path, subfolder=self._subfolder).eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    # Flux2VAE fully overrides encode()/decode() because it uses
    # patchify + BatchNorm instead of the standard shift/scale normalization.
    # See src/stage1/vae.py for the full implementation.
```

**When to fully override encode/decode:** Only when the normalization scheme is fundamentally different (e.g. BatchNorm, patchification). For simpler cases (just a different diffusers class or tensor format), prefer the hook methods.

## After creating the class

1. Export from `src/stage1/__init__.py`:
   ```python
   from .vae import VAE, Flux2VAE, QwenVAE, MyNewVAE
   __all__ = ["RAE", "VAE", "Flux2VAE", "QwenVAE", "MyNewVAE", "PixelEncoder"]
   ```

2. Set `target: stage1.MyNewVAE` in the eval config YAML (not `stage1.VAE`). The test script (`skills/stage1-add-vae/scripts/test_reconstruction.py`) auto-detects the class from the eval config.
