# Contributors

**Code Development.** Jaskirat Singh led most of the development for the unified codebase. Xingjian Leng added online T2I evaluation suites, REG and pixel-space methods. The individual contributions are:

- *Jaskirat Singh.* Designed the overall unified code structure and project direction using the same model architecture across various diffusion training tasks. Added stage1 (VAE/RAE) and stage2 (diffusion model) training across different tasks (ImageNet, T2I), 80+ different vision encoders for RAE and VAE, autoguidance (RAE), REPA, unified dataloader, in-context conditioning, MeanFlow, Gmuon optimiser, online gFID/rFID evaluation, simple T2I (using 256 text embedding tokens for T2I instead of 8 class condition tokens in ImageNet).
- *Xingjian Leng.* Helped add online evaluation for T2I experiments (GenEval, DPG-Bench, and GenAIBench). He also added REG and pixel-space implementation, helped co-add MeanFlow, and ran final experiments/results reported in the preliminary technical report.
