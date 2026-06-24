"""End-to-end training benchmark script for Stage 2 flow matching.

Runs training for a fixed number of steps and reports iterations/second
averaged over the last N steps (after warmup).

Usage:
    torchrun --nproc_per_node=8 src/train_benchmark.py \
        --config path/to/config.yaml \
        --total-steps 500 --warmup-steps 300 --precision bf16
"""

import argparse
import dataclasses
import math
import os
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from copy import deepcopy

import torch.distributed as dist
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from configs.stage2 import Stage2Config
from data import prepare_unified_dataloader
from encoders.vision_encoder import load_encoders
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport
from stage2.utils import encode_text, get_null_cond, setup_text_encoder, validate_stage2_config
from utils.dist_utils import cleanup_distributed, main_process_first, setup_distributed
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.train_utils import center_crop_arr, get_autocast_kwargs, update_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Stage-2 training throughput.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true", help="torch.compile the training loss function")
    parser.add_argument("--total-steps", type=int, default=500, help="Total training steps to run")
    parser.add_argument("--warmup-steps", type=int, default=300, help="Warmup steps before measuring throughput")
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.warmup_steps < args.total_steps, "warmup-steps must be less than total-steps"
    measure_steps = args.total_steps - args.warmup_steps

    #########################################################
    # Distributed + config setup
    #########################################################
    rank, world_size, device = setup_distributed()
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config))
    )
    config.post_process()
    validate_stage2_config(config)

    seed = config.training.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    autocast_kwargs = get_autocast_kwargs(args)

    if rank == 0:
        print(f"=== Benchmark Config ===")
        print(f"  total_steps:   {args.total_steps}")
        print(f"  warmup_steps:  {args.warmup_steps}")
        print(f"  measure_steps: {measure_steps}")
        print(f"  world_size:    {world_size}")
        print(f"  precision:     {args.precision}")
        print(f"  compile:       {args.compile}")

    #########################################################
    # Data setup
    #########################################################
    global_batch_size = config.training.global_batch_size or (
        config.training.batch_size * world_size * config.training.grad_accum_steps
    )
    assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    micro_batch_size = global_batch_size // (world_size * config.training.grad_accum_steps)

    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.training.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    needs_transform = config.dataset.type not in ("hf", "wds")

    dataloader = prepare_unified_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=micro_batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        transform=stage2_transform if needs_transform else None,
        condition_type=config.conditioning.type,
        virtual_epoch_steps=config.training.virtual_epoch_steps,
    )

    #########################################################
    # Models setup
    #########################################################
    latent_size = tuple(config.misc.latent_size)

    # stage1: rae - frozen
    rae: RAE = instantiate_from_config(config.stage_1).to(device)
    rae.eval()

    # repa target encoder
    repa_target_encoder = None
    if config.repa.use_repa:
        with main_process_first(rank):
            repa_target_encoder = load_encoders(config.repa.target_encoder, device, config.repa.target_encoder_resolution)[0]
        repa_target_encoder.eval()
        repa_target_encoder.model.requires_grad_(False)
        config.repa.z_dim = repa_target_encoder.embed_dim

    # text encoder
    text_encoder = setup_text_encoder(config, rank, device)

    config.prepare_model_params()

    # stage2: model - trainable
    model: Stage2ModelProtocol = instantiate_from_config(config.stage_2).to(device)
    model.requires_grad_(True)
    # stage2 ema model
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()

    # ddp wrapper
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    model = ddp_model.module
    ddp_model.train()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  model_params:  {total_params:.2f}M")
        print(f"  trainable:     {trainable_params:.2f}M")
        print(f"  batch/gpu:     {micro_batch_size}")
        print(f"  global_batch:  {global_batch_size}")
        print(f"========================")

    #########################################################
    # Optimizer + Scheduler setup
    #########################################################
    optimizer, _ = build_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        config.training.optimizer,
    )

    steps_per_epoch = len(dataloader) // config.training.grad_accum_steps

    scheduler = None
    if config.training.scheduler is not None:
        scheduler, _ = build_scheduler(optimizer, steps_per_epoch, config.training.scheduler)

    #########################################################
    # Transport setup
    #########################################################
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)

    if args.compile:
        transport.training_losses = torch.compile(transport.training_losses)

    # null conditions for CFG dropout
    model_kwargs_null = get_null_cond(
        text_encoder, config.conditioning.type, config.misc.num_classes, micro_batch_size, device
    )

    #########################################################
    # Training benchmark loop
    #########################################################
    dist.barrier()

    global_step = 0
    epoch = 0
    measure_start_time = None
    loss_history = []
    flops_per_step = 0
    flop_profiling = False
    flop_counted = False

    if rank == 0:
        print(f"\nStarting benchmark: {args.total_steps} steps ({args.warmup_steps} warmup + {measure_steps} measured)")

    optimizer.zero_grad()

    while global_step < args.total_steps:
        dataloader.set_epoch(epoch)

        for step, (images, y) in enumerate(dataloader):
            images = images.to(device)

            # Encode images to latents
            with torch.no_grad():
                z = rae.encode(images)
                if repa_target_encoder is not None:
                    raw_images = images.clone() * 255.0
                    raw_img_preprocessed = repa_target_encoder.preprocess(raw_images)
                    z_clean = repa_target_encoder.forward_features(raw_img_preprocessed)['x_norm_patchtokens']
                else:
                    z_clean = None

            # Encode conditions
            if config.conditioning.type == "text":
                context, context_attn_mask = encode_text(text_encoder, y)
            else:
                context, context_attn_mask = y.to(device), None

            # Forward + backward
            model_kwargs = dict(context=context, attn_mask=context_attn_mask)

            with autocast(**autocast_kwargs):
                loss_dict = transport.training_losses(
                    ddp_model, z, model_kwargs, model_kwargs_null,
                    z_clean=z_clean,
                    repa_coeff=config.repa.repa_coeff if config.repa.use_repa else None,
                    base_model_coeff=config.internal_guidance.base_model_coeff,
                    cfg_dropout_prob=config.conditioning.cfg_dropout_prob,
                )
                loss_diff = loss_dict["loss"].mean()
                loss_repa = loss_dict.get("loss_repa", torch.tensor(0.0, device=device)).mean()
                loss = loss_diff + loss_repa if config.repa.use_repa else loss_diff

            loss = loss / config.training.grad_accum_steps

            is_accum_step = (step + 1) % config.training.grad_accum_steps != 0
            if is_accum_step:
                with ddp_model.no_sync():
                    loss.backward()
            else:
                loss.backward()

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

                # Profile FLOPs on a single step right after warmup
                if global_step == args.warmup_steps - 1 and rank == 0 and not flop_counted:
                    try:
                        from torch.utils.flop_counter import FlopCounterMode
                        flop_counter = FlopCounterMode(display=False)
                        flop_counter.__enter__()
                        flop_profiling = True
                    except ImportError:
                        flop_profiling = False

                if global_step == args.warmup_steps and flop_profiling and rank == 0:
                    flop_counter.__exit__(None, None, None)
                    flops_per_step = flop_counter.get_total_flops()
                    flop_profiling = False
                    flop_counted = True
                    print(f"  FLOPs/step (rank 0): {flops_per_step / 1e9:.2f} GFLOPs")

                # Start measurement timer after warmup
                if global_step == args.warmup_steps:
                    torch.cuda.synchronize()
                    measure_start_time = time.perf_counter()
                    if rank == 0:
                        print(f"  Warmup complete at step {global_step}. Starting measurement...")

                loss_history.append(loss_diff.item())

                # Progress logging
                if rank == 0 and global_step % 50 == 0:
                    print(f"  Step {global_step}/{args.total_steps} | loss: {loss_diff.item():.4f}")

                if global_step >= args.total_steps:
                    break

        epoch += 1

    #########################################################
    # Compute and report results
    #########################################################
    torch.cuda.synchronize()
    measure_end_time = time.perf_counter()
    total_measure_time = measure_end_time - measure_start_time

    it_per_sec = measure_steps / total_measure_time
    sec_per_it = total_measure_time / measure_steps
    samples_per_sec = it_per_sec * global_batch_size

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"  Steps measured:      {measure_steps} (last {measure_steps} of {args.total_steps})")
        print(f"  Total measure time:  {total_measure_time:.2f}s")
        print(f"  Iterations/sec:      {it_per_sec:.4f}")
        print(f"  Seconds/iteration:   {sec_per_it:.4f}")
        print(f"  Samples/sec:         {samples_per_sec:.2f} (global_batch_size={global_batch_size})")
        if loss_history:
            print(f"  Final loss:          {loss_history[-1]:.6f}")
            n = min(100, len(loss_history))
            print(f"  Avg loss (last {n}):  {sum(loss_history[-n:]) / n:.6f}")
        if flops_per_step > 0:
            gflops_per_step = flops_per_step / 1e9
            tflops_per_sec = (flops_per_step * it_per_sec * world_size) / 1e12
            print(f"  GFLOPs/step (rank0): {gflops_per_step:.2f}")
            print(f"  TFLOP/s (total):     {tflops_per_sec:.2f}")
        print(f"{'='*50}")

    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
