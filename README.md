# diffusion-bench

```text
############################################################################
#                                                                            #
#   ____  _  __  __           _                            .-----------.     #
#  |  _ \(_)/ _|/ _|_   _ ___(_) ___  _ __                 |           |     #
#  | | | | | |_| |_| | | / __| |/ _ \| '_ \                | ░▒▓█▓▒░▒▓ |     #
#  | |_| | |  _|  _| |_| \__ \ | (_) | | | |               | ▒▓█████▓▒ |     #
#  |____/|_|_| |_|  \__,_|___/_|\___/|_| |_|               | ▓███████▓ |     #
#                                                          |     ↓     |     #
#   ____                  _                                | █████████ |     #
#  | __ )  ___ _ __   ___| |__                             | ▓███████▓ |     #
#  |  _ \ / _ \ '_ \ / __| '_ \                            | ▒▓█████▓▒ |     #
#  | |_) |  __/ | | | (__| | | |                           |           |     #
#  |____/ \___|_| |_|\___|_| |_|                           '-----------'     #
#                                                                            #
#           Because ImageNet evaluation alone is no longer enough!           #
#                                                                            #
############################################################################
```

[![Discord](https://img.shields.io/badge/Join_our_discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/jh5Bz8uHEr) [![GitHub](https://img.shields.io/badge/diffusion--bench-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/End2End-Diffusion/diffusion-bench) [![Blog](https://img.shields.io/badge/Blog-105864?style=for-the-badge&logo=readthedocs&logoColor=white)](https://end2end-diffusion.github.io/blog/posts/diffusion-bench/)

> **📣 Announcement post**: [Call for DiffusionBench: A Holistic Benchmark for Diffusion Transformers](https://end2end-diffusion.github.io/blog/posts/diffusion-bench/). Help us grow the benchmark with new evaluation axes, new metrics, and faithful reproductions of published methods.

This repo contains the unified codebase for DiffusionBench. It supports training and evaluation across different generation tasks (ImageNet, T2I, ...) through a single interface. Please see the sections below for the detailed structure. Come join us!

## Quickstart

### Setup

```sh
# install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# prepare data
uv run python scripts/prepare.py --data {all,imagenet,t2i,eval}

# download pretrained models
uv run hf download diffusion-bench/diffusion-bench --local-dir pretrained_models --exclude .gitattributes
```

### Training

Reproduction flow: **Stage 1 → Stage 2**. Set these environment variables first (used for the output directory and W&B logging):

```sh
export EXPERIMENT_NAME=<run-name>
export ENTITY=<wandb-entity>
export PROJECT=<wandb-project>
export WANDB_KEY=<key>
```

**Stage 1.** Train the RAE tokenizer:
```sh
uv run torchrun --standalone --nproc_per_node=8 \
    src/train_stage1.py \
    --config [STAGE1_CONFIG_PATH] \
    --results-dir results/stage1 --precision bf16 --compile --wandb
```

**Stage 2.** Train the diffusion model on VAE/RAE/Pixel space:
```sh
uv run torchrun --standalone --nproc_per_node=8 \
    src/train.py \
    --config [STAGE2_CONFIG_PATH] \
    --results-dir results/stage2 --precision bf16 --compile --wandb
```

### Evaluation

Stage 2 *training* configs run online evaluation during training (the `eval:` block). For standalone evaluation of a released checkpoint, use the **`sampling/`** configs — each embeds `stage_2.ckpt` (pointing into `pretrained_models/`) and the eval-time guidance, so the weights load automatically:
```sh
export EXPERIMENT_NAME=<run-name>

# stage 1 reconstruction (rFID/PSNR/SSIM/LPIPS)
uv run torchrun --nproc_per_node=8 src/offline_eval_stage1.py --config [STAGE1_CONFIG_PATH]

# stage 2 generation (FID/IS, GenEval/DPGBench/...)
uv run torchrun --nproc_per_node=8 src/offline_eval.py --config [STAGE2_CONFIG_PATH]
```

### Available Configs

```
configs/
├── stage1/
└── stage2/
    ├── training/
    │   ├── imagenet/
    │   └── t2i/
    └── sampling/
        ├── imagenet/
        └── t2i/
```

Stage 2 spans VAE (11), RAE (6), REG (4), and Pixel (3) families, identical across ImageNet and T2I. Swap any config between tasks with a single path change. The `sampling/` set mirrors `training/` but adds the trained checkpoint and eval-time guidance, so it runs offline eval directly.

For ImageNet, pick the CFG-off baseline (`[STAGE2_CONFIG_PATH].yaml`) or the per-model best-CFG variant (`[STAGE2_CONFIG_PATH]-cfg<scale>-t0.0-0.9.yaml`).

## Supported Methods

| Category | Methods |
|----------|---------|
| **Latent Space** | `Pixel Space` <br>`RAE` (30+ representation encoders): `DINOv2` `SigLIP2` `WebSSL` `PE` `LangPE` and more <br>`RAEv2` (30+ representation encoders): `DINOv2` `SigLIP2` `WebSSL` `PE` `LangPE` etc <br>`VAE` (10+ VAEs): `FLUX.2` `FLUX.1` `SD3.5` `VA-VAE` `E2E-VAE` and more |
| **Output Prediction** | `x-prediction` `v-prediction` |
| **Transport** | `Rectified-Flow` `MeanFlow` `Improved-MeanFlow` `Pixel-MeanFlow` `Drifting` |
| **Loss** | `Flow Matching` `REPA` `iREPA` |
| **Architecture** | `LightningDiT` `JiT` `DDT` |
| **Tasks** | `ImageNet`: class-conditional generation <br>`T2I`: text-to-image generation |
| **Evaluation** | ImageNet: `FID` `IS` <br>T2I: `GenEval` `DPGBench` `GenAIBench` `SimpleEval` `VQAScore` |
| **Training Backend** | `DDP` `FSDP [TODO]` |

## Compatibility

| | Status | Details |
|--|--------|---------|
| **Coding Agents** | Yes | Agent-compatible. See [`skills/`](skills/) for setup and workflow skills. |
| **AutoResearch** | Yes | AutoResearch-compatible. See the `nanogen/autoresearch` branch. |

## Contributing

We welcome contributions! Please refer to [`docs/contributors.md`](docs/contributors.md) and [`docs/contributing.md`](docs/contributing.md) for further details.

## Acknowledgments
The codebase is built upon some amazing projects:
- [RAE](https://github.com/bytetriper/RAE)
- [RAEv2](https://github.com/nanovisionx/RAEv2)
- [REPA](https://github.com/sihyun-yu/REPA)
- [REPA-E](https://github.com/End2End-Diffusion/REPA-E)
- [JiT](https://github.com/LTH14/JiT)

We thank the authors for making their work publicly available.
