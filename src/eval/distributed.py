"""Shared distributed eval infrastructure (sharding, NCCL gather to rank 0)."""

from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset


def create_eval_dataloader(dataset, rank: int, world_size: int,
                           num_samples: int, batch_size: int) -> DataLoader:
    """Shard dataset across ranks and return a DataLoader for this rank's subset."""
    N = min(len(dataset), num_samples)
    chunk = N // world_size

    if rank < world_size - 1:
        start = rank * chunk
        end = (rank + 1) * chunk
    else:
        start = rank * chunk
        end = N

    rank_indices = list(range(start, end))
    subset = Subset(dataset, rank_indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context="spawn",
    )


def gather_to_rank0(local: np.ndarray, rank: int, world_size: int, device,
                    num_samples: int, chunk: int = 256) -> Optional[np.ndarray]:
    """Collective: gather every rank's uint8 array to rank 0 over NCCL (no shared disk).

    Returns rank-major concatenation truncated to num_samples on rank 0, None elsewhere.
    Ranks may hold different counts; data moves in `chunk`-sized pieces to bound GPU memory.
    """
    n = torch.tensor([len(local)], device=device)
    sizes = [torch.zeros_like(n) for _ in range(world_size)]
    dist.all_gather(sizes, n)
    sizes = [int(s) for s in sizes]
    parts = [[] for _ in range(world_size)]
    for start in range(0, max(sizes), chunk):
        end = min(start + chunk, max(sizes))
        buf = torch.zeros((end - start, *local.shape[1:]), dtype=torch.uint8, device=device)
        k = max(0, min(end, len(local)) - start)
        if k:
            buf[:k] = torch.from_numpy(local[start:start + k]).to(device)
        bufs = [torch.empty_like(buf) for _ in range(world_size)] if rank == 0 else None
        dist.gather(buf, bufs, dst=0)
        if rank == 0:
            for r in range(world_size):
                kr = max(0, min(end, sizes[r]) - start)
                if kr:
                    parts[r].append(bufs[r][:kr].cpu().numpy())
    if rank != 0:
        return None
    return np.concatenate([p for rank_parts in parts for p in rank_parts], axis=0)[:num_samples]
