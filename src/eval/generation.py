"""Distributed generation evaluation — FID, CLIPScore, VQAScore, GenEval, GenEval2, DPG-Bench,
PRISM-Bench, Qwen-Image-Bench."""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm

from .clipscore import CLIPScoreEvaluator
from .distributed import create_eval_dataloader, gather_to_rank0
from .distributional import compute_distributional_metrics, filter_distributional
from .dpgbench import DPGEvaluator
from .fid import calculate_fd_dinov2, calculate_gfid
from .geneval import GenEvalEvaluator
from .geneval2 import GenEval2Evaluator
from .prism_bench import PrismBenchEvaluator
from .qwen_image_bench import QwenImageBenchEvaluator
from .vqascore import VQAScoreEvaluator


def _requested_submetrics(metrics_to_compute: List[str], prefix: str, valid: tuple) -> list:
    """Bare `prefix` requests every sub-metric; `prefix_<sub>` requests one.

    Returned in `valid` order so every rank builds local_scores in the same key
    order (the all-reduce in _aggregate_distributed_metrics is positional).
    """
    requested = set()
    for elem in metrics_to_compute:
        if elem == prefix:
            requested.update(valid)
        elif elem.startswith(prefix + '_'):
            sub = elem[len(prefix) + 1:]
            if sub in valid:
                requested.add(sub)
            else:
                print(f"[{prefix}] Warning: unknown metric '{elem}', skipping. Valid: {valid}")
    return [sub for sub in valid if sub in requested]


def _accumulate_submetrics(local_scores: dict, prefix: str, batch_scores: Dict[str, torch.Tensor]):
    """Add per-image sub-metric scores, skipping NaNs (sparse dimensions / unscored images)."""
    for sub, sub_scores in batch_scores.items():
        key = f'{prefix}_{sub}'
        if key in local_scores:
            valid = ~torch.isnan(sub_scores)
            local_scores[key]['sum'] += sub_scores[valid].sum().item()
            local_scores[key]['count'] += int(valid.sum().item())


def _init_evaluators(metrics_to_compute: List[str], condition_type: str, device: torch.device):
    """Initialize metric evaluators and local score accumulators."""
    evaluators = {}
    local_scores = {}

    if 'clipscore' in metrics_to_compute and condition_type == 'text':
        evaluators['clipscore'] = CLIPScoreEvaluator(device=str(device))
        local_scores['clipscore'] = {'sum': 0.0, 'count': 0}

    if any(elem.startswith('vqascore') for elem in metrics_to_compute) and condition_type == 'text':
        vqascore_models = [elem for elem in metrics_to_compute if elem.startswith('vqascore')]
        vqascore_evaluators = {}
        for model_name in vqascore_models:
            model_name_ = model_name.split('_')[-1] if '_' in model_name else 'clip-flant5-xl'
            vqascore_evaluators[model_name] = VQAScoreEvaluator(model_name=model_name_, device=str(device))
            local_scores[model_name] = {'sum': 0.0, 'count': 0}
        evaluators['vqascore'] = vqascore_evaluators

    if 'geneval' in metrics_to_compute and condition_type == 'text':
        evaluators['geneval'] = GenEvalEvaluator(device=str(device))
        local_scores['geneval'] = {'sum': 0.0, 'count': 0}

    if 'dpgbench' in metrics_to_compute and condition_type == 'text':
        evaluators['dpgbench'] = DPGEvaluator(device=str(device))
        local_scores['dpgbench'] = {'sum': 0.0, 'count': 0}

    # GenEval2 / Qwen-Image-Bench: one judge pass yields all sub-metrics, reported as
    # '<name>_<sub>'. Bare '<name>' reports all of them, '<name>_<sub>' just one.
    if condition_type == 'text':
        for name, cls in (('geneval2', GenEval2Evaluator), ('qwen_image_bench', QwenImageBenchEvaluator)):
            requested = _requested_submetrics(metrics_to_compute, name, cls.METRICS)
            if requested:
                evaluators[name] = cls(device=str(device))
                for sub in requested:
                    local_scores[f'{name}_{sub}'] = {'sum': 0.0, 'count': 0}

    # PRISM-Bench: the metric string picks the judge (like VQAScore). Bare 'prism_bench'
    # uses the package default; 'prism_bench_<judge>' (e.g. prism_bench_qwen2.5-vl-7b)
    # picks another. Each judge reports every sub-metric as '<metric>_<sub>'.
    prism_metrics = [m for m in metrics_to_compute if m == 'prism_bench' or m.startswith('prism_bench_')]
    if prism_metrics and condition_type == 'text':
        evaluators['prism_bench'] = {}
        for elem in prism_metrics:
            judge = None if elem == 'prism_bench' else elem[len('prism_bench_'):]
            evaluators['prism_bench'][elem] = PrismBenchEvaluator(model_name=judge, device=str(device))
            for sub in PrismBenchEvaluator.METRICS:
                local_scores[f'{elem}_{sub}'] = {'sum': 0.0, 'count': 0}

    return evaluators, local_scores


def _aggregate_distributed_metrics(local_scores: dict, device: torch.device) -> Dict[str, float]:
    """All-reduce local score sums/counts across ranks and return averaged metrics."""
    metrics = {}
    # Sorted: all_reduce is positional, so every rank must reduce keys in the same order.
    for metric_name, scores in sorted(local_scores.items()):
        sum_tensor = torch.tensor([scores['sum']], device=device, dtype=torch.float64)
        count_tensor = torch.tensor([scores['count']], device=device, dtype=torch.float64)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        if count_tensor.item() > 0:
            metrics[metric_name] = sum_tensor.item() / count_tensor.item()
    return metrics


@torch.no_grad()
def evaluate_generation_distributed(
    model_fn,
    sample_fn,
    latent_size,
    additional_model_kwargs,
    use_guidance: bool,
    rae,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    experiment_dir: str,
    global_step: int,
    autocast_kwargs: dict,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    shared_tmpdir: Optional[str] = None,
    condition_type: str = "label",
    null_label: int = 1000,
    text_encoder=None,
    metrics_to_compute: Optional[List[str]] = None,
    cls_dim: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """
    Evaluate generation metrics using all GPUs in a distributed manner.

    Args:
        model_fn: Model forward function
        sample_fn: Sampling function
        latent_size: Shape of latent noise
        additional_model_kwargs: Additional kwargs for model forward
        use_guidance: Whether to use classifier-free guidance
        rae: RAE model for decoding latents to images
        val_dataset: Validation dataset (returns (image, label) or (image, text))
        num_samples: Number of samples to generate
        batch_size: Batch size per GPU for generation
        rank: Current GPU rank
        world_size: Total number of GPUs
        device: Device to use
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path to existing reference NPZ file
        shared_tmpdir: Unused (kept for config compatibility); samples are gathered over NCCL
        condition_type: Type of conditioning - "label" or "text"
        null_label: Null label index for CFG (label conditioning only)
        text_encoder: Text encoder for text conditioning (required if condition_type="text")
        metrics_to_compute: List of metrics to compute (default: ['fid'])

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    if rank == 0:
        print(f"\n[Eval] Starting distributed sampling evaluation at step {global_step}")
    loader = create_eval_dataloader(val_dataset, rank, world_size, num_samples, batch_size)

    # Initialize evaluators
    if metrics_to_compute is None:
        metrics_to_compute = ['fid']
    evaluators, local_scores = _init_evaluators(metrics_to_compute, condition_type, device)

    # Generate images on this rank
    generations = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Sampling", file=sys.stdout) if rank == 0 else loader

    with torch.inference_mode():
        for _, cond in iterator:
            # Handle conditioning based on type
            if condition_type == "text":
                n = len(cond)
                z = torch.randn(n, *latent_size, device=device)
                enc_out = text_encoder(list(cond))
                context = enc_out["tokens"]
                context_attn_mask = enc_out["attention_mask"]
                if use_guidance:
                    z = torch.cat([z, z], dim=0)
                    enc_null = text_encoder([""] * n)
                    context_null = enc_null["tokens"]
                    context_attn_mask_null = enc_null["attention_mask"]
                    context = torch.cat([context, context_null], dim=0)
                    context_attn_mask = torch.cat([context_attn_mask, context_attn_mask_null], dim=0)
            else:
                n = cond.size(0)
                z = torch.randn(n, *latent_size, device=device)
                context = cond.to(device)
                context_attn_mask = None
                if use_guidance:
                    z = torch.cat([z, z], dim=0)
                    context_null = torch.full((n,), null_label, device=device)
                    context = torch.cat([context, context_null], dim=0)

            model_kwargs = dict(context=context, attn_mask=context_attn_mask, **additional_model_kwargs)
            if cls_dim is not None:
                cls_init = torch.randn(n, cls_dim, device=device)
                model_kwargs["cls_t"] = torch.cat([cls_init, cls_init], dim=0) if use_guidance else cls_init
            with autocast(**autocast_kwargs):
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if use_guidance:
                    samples = samples.chunk(2, dim=0)[0]
                samples = rae.decode(samples).clamp(0, 1)
            gen_np = samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Compute distributed metrics during generation
            if 'clipscore' in evaluators:
                batch_scores = evaluators['clipscore'].compute_batch_scores(gen_np, list(cond))
                local_scores['clipscore']['sum'] += batch_scores.sum().item()
                local_scores['clipscore']['count'] += len(cond)
            if 'vqascore' in evaluators:
                for model_name, evaluator in evaluators['vqascore'].items():
                    batch_scores = evaluator.compute_batch_scores(gen_np, list(cond))
                    local_scores[model_name]['sum'] += batch_scores.sum().item()
                    local_scores[model_name]['count'] += len(cond)
            if 'geneval' in evaluators:
                batch_scores = evaluators['geneval'].compute_batch_scores(gen_np, list(cond))
                local_scores['geneval']['sum'] += batch_scores.sum().item()
                local_scores['geneval']['count'] += len(cond)
            if 'dpgbench' in evaluators:
                batch_scores = evaluators['dpgbench'].compute_batch_scores(gen_np, list(cond))
                local_scores['dpgbench']['sum'] += batch_scores.sum().item()
                local_scores['dpgbench']['count'] += len(cond)
            for name in ('geneval2', 'qwen_image_bench'):
                if name in evaluators:
                    batch_scores = evaluators[name].compute_batch_scores(gen_np, list(cond))
                    _accumulate_submetrics(local_scores, name, batch_scores)
            if 'prism_bench' in evaluators:
                for elem, evaluator in evaluators['prism_bench'].items():
                    batch_scores = evaluator.compute_batch_scores(gen_np, list(cond))
                    _accumulate_submetrics(local_scores, elem, batch_scores)

            for img in gen_np:
                generations.append(img)

    generations = np.stack(generations)

    # Wait for all ranks to finish generation
    dist.barrier()

    # Distributed metrics: all_reduce sum and count across all ranks
    metrics = _aggregate_distributed_metrics(local_scores, device)

    # FID etc. need every sample on rank 0: gather over NCCL (collective, so all ranks call it).
    distributional = filter_distributional(metrics_to_compute)
    need_combined = (
        'fid' in metrics_to_compute
        or 'inception_score' in metrics_to_compute
        or bool(distributional)
    )
    combined_recons = (
        gather_to_rank0(generations, rank, world_size, device, num_samples) if need_combined else None
    )

    # Rank 0 computes FID
    if rank == 0:
        if need_combined:
            print(f"[Eval] Combined generation NPZ shape: {combined_recons.shape}")

            native = {'fid', 'inception_score', 'fd_dinov2'} & set(metrics_to_compute)
            device_str = "cuda" if device.type == "cuda" else "cpu"
            if native:
                if not os.path.exists(reference_npz_path):
                    raise FileNotFoundError(f"Reference NPZ not found at {reference_npz_path}")
                ref_stats = np.load(reference_npz_path)
                print(f"[Eval] Loaded reference NPZ from {reference_npz_path}")
            if 'fid' in native or 'inception_score' in native:
                fid, inception_score = calculate_gfid(combined_recons, ref_stats, metric_batch_size, device_str)
                metrics['fid'] = fid
                metrics['inception_score'] = inception_score
            if 'fd_dinov2' in native:  # needs dino_mu/dino_sigma in the reference (e.g. GPIC stats)
                metrics['fd_dinov2'] = calculate_fd_dinov2(combined_recons, ref_stats, metric_batch_size, device_str)

            # fd_evaluator distributional metrics (fdr6/mind6/...) on the same gathered array.
            # fid/inception_score/fd_dinov2 stay on the native path above, so exclude them.
            fd_metrics = [m for m in distributional if m not in native]
            if fd_metrics:
                metrics.update(compute_distributional_metrics(
                    combined_recons, fd_metrics, device=device, batch_size=metric_batch_size))

        # Print results
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

    dist.barrier()
    return metrics if metrics else None
