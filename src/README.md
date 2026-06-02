# src/

Core training and inference code for RAEv2.

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `train_stage1.py` | Train Stage-1 RAE decoder (frozen encoder + trainable decoder) |
| `train.py` | Train Stage-2 diffusion transformer on RAE latents |
| `train_e2e.py` | End-to-end training (encoder + decoder + diffusion) |
| `offline_eval.py` | Run evaluation metrics offline |

---

## Folders

| Folder | Purpose |
|--------|---------|
| `stage1/` | RAE model definitions (encoder-decoder autoencoder) |
| `stage2/` | Diffusion transformer models (DiT variants) |
| `encoders/` | Pretrained encoder wrappers (DINOv2, SigLIP2, MAE, etc.) used for both RAE and REPA w/ custom encoders - so far 30+ encoders supported. Contains `models/` submodule for low-level architectures. |
| `data/` | Dataset classes and dataloaders |
| `eval/` | Evaluation metrics (FID, CLIP score, linear probe, etc.) |
| `disc/` | Discriminator and GAN loss for Stage-1 training |
| `sample/` | Sampling/inference scripts for Stage-1 and Stage-2 |
| `utils/` | Shared utilities (checkpointing, distributed, logging, etc.) |

---

## Guidelines

- **Entry points stay minimal** - logic lives in submodules

See `/GUIDELINE.md` for full coding standards.
