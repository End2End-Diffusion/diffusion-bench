"""Stage 2 training engine: train_one_epoch and helpers."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.distributed as dist
import wandb
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.stage2 import Stage2Config
from stage2.utils import (
    encode_text,
    get_fixed_viz_batch_conditions,
    get_null_cond,
    sample_and_decode,
)
from utils import wandb_utils
from utils.checkpoint import save_stage2_checkpoint
from utils.guidance_utils import get_model_forward_fn
from utils.logging import save_eval_to_csv
from utils.sync_utils import sync_checkpoint_async, sync_evals_async
from utils.train_utils import update_ema

logger = logging.getLogger("rae")


#########################################################
# Main training function
#########################################################
def train_one_epoch(
    *, # * forces all arguments to be passed as keyword arguments
    ddp_model: DDP,
    ema_model: torch.nn.Module,
    rae,
    percep_loss,
    transport,
    eval_sampler,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    autocast_kwargs: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: Stage2Config,
    args,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    checkpoint_dir: str,
    experiment_dir: str,
    progress_bar,
    text_encoder=None,
    repa_target_encoder=None,
    eval_datasets: Optional[Dict] = None,
    viz_fixed: Optional[Dict] = None,
) -> int:
    """Run one epoch of Stage 2 training. Returns updated global_step.

    Args:
        viz_fixed: Mutable dict with keys 'zs', 'y', 'encoder_hidden_states',
            'encoder_attention_mask'. Populated from first batch, persists across epochs.
    """
    #########################################################
    # Setup
    #########################################################
    model = ddp_model.module

    # Guidance: derive model_fn / ema_model_fn / sample_kwargs from config
    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    ema_model_fn, _ = get_model_forward_fn(ema_model, config.guidance)
    use_guidance = config.guidance.any_guidance_active

    # Eval settings
    do_eval = config.eval is not None and eval_datasets is not None
    if do_eval: eval_dir = config.eval.eval_dir
    experiment_name = os.environ.get("EXPERIMENT_NAME")

    # Get null conditions for CFG dropout
    model_kwargs_null = get_null_cond(text_encoder, config.conditioning.type, config.misc.num_classes, micro_batch_size, device)

    # per-epoch state
    num_viz_samples = viz_fixed['zs'].shape[0] if viz_fixed is not None else 0
    epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
    num_batches = 0
    optimizer.zero_grad()

    # save checkpoint at epoch start
    if config.training.checkpoint_interval > 0 and epoch % config.training.checkpoint_interval == 0 and rank == 0:
        logger.info(f"Saving checkpoint at epoch {epoch}...")
        ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
        save_stage2_checkpoint(ckpt_path, global_step, epoch, ddp_model, ema_model, optimizer, scheduler)
        if args.sync_checkpoints:
            sync_checkpoint_async(checkpoint_dir, logger)
            if do_eval: sync_evals_async(eval_dir, logger)

    #########################################################
    # Training loop
    #########################################################
    dataloader.set_epoch(epoch)
    for step, (images, y) in enumerate(dataloader):
        images = images.to(device)

        # Encode images to latents and compute REPA targets
        with torch.no_grad():
            z = rae.encode(images)
            z_clean = cls_clean = None
            if repa_target_encoder is not None:
                raw_images = images.clone() * 255.0
                raw_img_preprocessed = repa_target_encoder.preprocess(raw_images)
                feats = repa_target_encoder.forward_features(raw_img_preprocessed)
                z_clean = feats['x_norm_patchtokens']
                if config.repa.use_reg:
                    cls_clean = feats['x_norm_clstoken']
                    if config.repa.use_repa:
                        z_clean = torch.cat([cls_clean.unsqueeze(1), z_clean], dim=1)

        # Capture fixed conditions from first batch
        if viz_fixed is not None:
            viz_fixed = get_fixed_viz_batch_conditions(viz_fixed, y, config.conditioning.type, text_encoder, device)

        # Encode conditions
        if config.conditioning.type == "text":
            context, context_attn_mask = encode_text(text_encoder, y)
        else:
            context, context_attn_mask = y.to(device), None

        #########################################################
        # Forward + backward
        #########################################################
        model_kwargs = dict(context=context, attn_mask=context_attn_mask)

        with autocast(**autocast_kwargs):
            loss_dict = transport.training_losses(
                ddp_model, z, model_kwargs, model_kwargs_null,
                percep_loss=percep_loss,
                z_clean=z_clean,
                repa_coeff=config.repa.repa_coeff if config.repa.use_repa else None,
                base_model_coeff=config.internal_guidance.base_model_coeff,
                cfg_dropout_prob=config.conditioning.cfg_dropout_prob,
                ema_model=ema_model,
                cls_clean=cls_clean,
                reg_coeff=config.repa.reg_coeff if config.repa.use_reg else None,
            )
            loss_diff = loss_dict["loss"].mean()
            loss_percep = loss_dict.get("loss_percep", torch.tensor(0.0, device=device)).mean()
            loss = loss_diff + loss_percep

            loss_repa = loss_dict.get("loss_repa", torch.tensor(0.0, device=device)).mean()
            loss_reg = loss_dict.get("loss_reg", torch.tensor(0.0, device=device)).mean()
            if config.repa.use_repa:
                loss = loss + loss_repa
            if config.repa.use_reg:
                loss = loss + loss_reg

        loss = loss / config.training.grad_accum_steps

        is_accum_step = (step + 1) % config.training.grad_accum_steps != 0
        if is_accum_step:
            with ddp_model.no_sync():
                loss.backward()
        else:
            loss.backward()  # DDP auto-syncs gradients on final micro-step

        # Step optimizer and scheduler at grad accumulation boundary
        if not is_accum_step:
            transport.post_backward(ddp_model)
            if config.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config.training.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            update_ema(ema_model, ddp_model.module, decay=config.training.ema_decay)
            global_step += 1
            progress_bar.update(1)

        epoch_metrics['loss'] += loss_diff.detach()
        num_batches += 1

        # Skip logging/viz/eval on non-boundary micro-steps
        if is_accum_step:
            continue

        #########################################################
        # Logging and visualization
        #########################################################
        if config.training.log_interval > 0 and global_step % config.training.log_interval == 0 and rank == 0:
            cur_loss = loss_diff.item()
            stats = {"train/loss": cur_loss, "train/lr": optimizer.param_groups[0]["lr"]}
            if config.repa.use_repa:
                stats["train/loss_repa"] = loss_repa.item()
            if config.repa.use_reg:
                stats["train/loss_reg"] = loss_reg.item()
            if "loss_base" in loss_dict:
                stats["train/loss_base"] = loss_dict["loss_base"].mean().item()
            if "loss_percep" in loss_dict:
                stats["train/loss_percep"] = loss_dict["loss_percep"].mean().item()
            if "orig_loss_base" in loss_dict:
                stats["train/orig_loss_base"] = loss_dict["orig_loss_base"].mean().item()
            if "orig_mf_loss" in loss_dict:
                stats["train/orig_mf_loss"] = loss_dict["orig_mf_loss"].mean().item()
            if "dudt_norm" in loss_dict:
                stats["train/dudt_norm"] = loss_dict["dudt_norm"].mean().item()
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
            )
            if args.wandb:
                wandb_utils.log(stats, step=global_step)
            progress_bar.set_postfix(loss=cur_loss, lr=optimizer.param_groups[0]["lr"])

        # Sampling visualization
        if global_step % config.training.sample_every == 0:
            model.eval()
            logger.info("Generating EMA samples...")
            sample_args = dict(
                eval_sampler=eval_sampler, model_fn=ema_model_fn,
                sample_model_kwargs=sample_model_kwargs, rae=rae,
                use_guidance=use_guidance, condition_type=config.conditioning.type,
                text_encoder=text_encoder, num_classes=config.misc.num_classes,
                device=device, autocast_kwargs=autocast_kwargs,
            )
            if rank == 0:
                with torch.no_grad():
                    samples_dict = {}
                    # 1. Batch samples (from current batch conditions)
                    batch_n = min(num_viz_samples, context.shape[0])
                    zs_batch = torch.randn(batch_n, *config.misc.latent_size, device=device, dtype=torch.float32)
                    cls_init = torch.randn(batch_n, config.repa.z_dim, device=device, dtype=torch.float32) if config.repa.use_reg else None
                    samples_dict["samples/batch"] = sample_and_decode(
                        zs_batch, context[:batch_n],
                        context_attn_mask[:batch_n] if context_attn_mask is not None else None,
                        cls_t=cls_init,
                        **sample_args,
                    )
                    # 2. Fixed samples (consistent across epochs)
                    if viz_fixed is not None and viz_fixed['context'] is not None:
                        samples_dict["samples/fixed"] = sample_and_decode(
                            viz_fixed['zs'].clone(), viz_fixed['context'].clone(),
                            viz_fixed['attn_mask'].clone() if viz_fixed['attn_mask'] is not None else None,
                            cls_t=cls_init,
                            **sample_args,
                        )
                    if args.wandb: # log samples to wandb
                        for name, samples in samples_dict.items():
                            grid = wandb_utils.array2grid(samples)
                            wandb.log({name: wandb.Image(grid)}, step=global_step)
            dist.barrier()
            logger.info("Generating EMA samples done.")
            model.train() # set model back to train mode

        #########################################################
        # Evaluation; distributed evaluation
        #########################################################
        if do_eval and config.eval.eval_interval > 0 and global_step % config.eval.eval_interval == 0:
            from eval import evaluate_generation_distributed
            logger.info("Starting evaluation...")
            model.eval()
            # eval ema or both ema and running model if eval_model is True
            eval_models = [(ema_model_fn, "ema")] if not config.eval.eval_model else [(ema_model_fn, "ema"), (model_fn, "model")]
            for fn, mod_name in eval_models:
                for ds_name, ds_info in eval_datasets.items():
                    logger.info(f"Evaluating {mod_name} on {ds_name}...")
                    eval_stats = evaluate_generation_distributed(
                        fn, eval_sampler, tuple(config.misc.latent_size), sample_model_kwargs,
                        use_guidance, rae, ds_info.dataset, len(ds_info.dataset),
                        rank=rank, world_size=world_size, device=device,
                        batch_size=micro_batch_size, experiment_dir=experiment_dir,
                        global_step=global_step, autocast_kwargs=autocast_kwargs,
                        reference_npz_path=ds_info.reference_npz,
                        shared_tmpdir=config.dataset.shared_tmpdir,
                        condition_type=ds_info.condition_type,
                        null_label=config.misc.num_classes,
                        text_encoder=text_encoder if ds_info.condition_type == "text" else None,
                        metrics_to_compute=ds_info.metrics,
                        cls_dim=config.repa.z_dim if config.repa.use_reg else None,
                    )
                    if eval_stats is not None and rank == 0:
                        save_eval_to_csv(experiment_name, mod_name, global_step, {'dataset': ds_name, **eval_stats}, eval_dir)
                        if args.wandb:
                            wandb_utils.log({f"eval_{mod_name}/{k}_{ds_name}": v for k, v in eval_stats.items()}, step=global_step)
            model.train() # set model back to train mode
            logger.info("Evaluation done.")


    #########################################################
    # Epoch summary
    #########################################################
    if rank == 0 and num_batches > 0:
        avg_loss = epoch_metrics['loss'].item() / num_batches
        epoch_stats = {"epoch/loss": avg_loss}
        logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
        if args.wandb:
            wandb_utils.log(epoch_stats, step=global_step)

    return global_step
