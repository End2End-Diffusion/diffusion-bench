---
name: stage1-add-vae
description: Add a new HuggingFace-supported VAE to the stage1 tokenizer pipeline. Use when integrating a new VAE model from diffusers, adding autoencoder support, or the user says "add vae", "new vae", "integrate vae", "add autoencoder", "support X VAE".
---

# Add a New VAE to Stage 1

## Quick overview

Adding a VAE touches these files (all in deterministic locations):

| File | What to do |
|---|---|
| `src/stage1/vae.py` | Add `VAEConfig` entry to `VAE_CONFIGS` dict. If the VAE needs a custom diffusers class or non-4D tensors, add a subclass. |
| `src/stage1/__init__.py` | Export the new class (only if custom subclass created). |
| `experiments/<user>/jobs/stage1-eval/configs/<name>.yaml` | Create eval config from template. Find the user's experiment dir by looking at existing configs. |
| `experiments/<user>/jobs/stage1-eval/jobs/stage1-offline-eval-vae.sh` | Append to `CONFIGS` array + update `--array=` range. |
| `docs/stage1_section.md` | Add row to benchmark table with `- | - | -` for metrics. |

## Step 0: Investigate the VAE

Run this before writing any code:
```sh
uv run python skills/stage1-add-vae/scripts/investigate_vae.py --model <hf_model_path> --subfolder vae
```

This script automatically:
1. Tries loading with `AutoencoderKL`, then specialized classes (`AutoencoderKLQwenImage`, `AutoencoderKLWan`, `AutoencoderKLFlux2`)
2. Prints config attributes (latents_mean, latents_std, scaling_factor, etc.)
3. Probes 4D vs 5D tensor format
4. Tests a full encode-decode round trip
5. Prints a ready-to-paste `VAEConfig(...)` entry
6. Tells you exactly what you need: base class only, or custom subclass

## Step 1: Add `VAEConfig` entry

In `src/stage1/vae.py`, add to the `VAE_CONFIGS` dict. Use the output from the investigation script.

If there is also an end-to-end (E2E) fine-tuned variant (from REPA-E), add a second entry with `pretrained_path="REPA-E/e2e-<name>"`.

## Step 2: Custom class (only if needed)

**Skip this if the investigation script says "Standard AutoencoderKL — base VAE class works" and "4D tensors work".**

Otherwise, create a subclass of `VAE`. See [references/custom-classes.md](references/custom-classes.md) for the two real examples:

- **QwenVAE** — different diffusers class + 5D tensor unsqueeze/squeeze. Overrides `_load_vae`, `_vae_encode`, `_vae_decode`. Simple.
- **Flux2VAE** — different diffusers class + patchify + BatchNorm. Fully overrides `encode`/`decode`. Complex.

The base class has three hook methods to override without duplicating normalization logic:
- `_load_vae()` — use a different diffusers class
- `_vae_encode(x)` — wrap the raw VAE encode (e.g. unsqueeze/squeeze)
- `_vae_decode(z)` — wrap the raw VAE decode

If you create a custom class, also:
- Export it from `src/stage1/__init__.py`
- Set `target: stage1.MyNewVAE` (not `stage1.VAE`) in eval configs

## Step 3: Create eval config

Find the user's experiment directory by looking at existing eval configs:
```sh
ls experiments/*/jobs/stage1-eval/configs/
```

Copy the template from [references/eval-config-template.yaml](references/eval-config-template.yaml), replacing FIXME values. Create one per variant (base + e2e if applicable).

## Step 4: Update job script + docs

1. In the job script (`stage1-offline-eval-vae.sh`): append to `CONFIGS` array, update `--array=` range.
2. In `docs/stage1_section.md`: add row(s) to the benchmark table. Use `f{downsample}d{channels}` for latent format. Leave metrics as `-` until eval runs.

## Step 5: Verify with reconstruction test

```sh
PYTHONPATH=src:$PYTHONPATH uv run python skills/stage1-add-vae/scripts/test_reconstruction.py --vae-type <name> --num-images 4
```

This loads the VAE, reconstructs a few ImageNet val images, prints per-image PSNR/SSIM, and saves side-by-side comparison images. Useful for catching issues before running the full 50K eval.

## References

- [references/custom-classes.md](references/custom-classes.md) — hook methods, QwenVAE and Flux2VAE examples
- [references/eval-config-template.yaml](references/eval-config-template.yaml) — copy-paste YAML template
- [references/normalization.md](references/normalization.md) — how shift/scale factors are resolved, debugging bad rFID
