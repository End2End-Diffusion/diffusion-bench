"""Unified dataloader interface for RAEv2 training."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .imagenet_hf_dataset import ImageNetHFDataset

T2I_HF_DATASETS = {'mscoco', 'mjhq', 'geneval', 'dpgbench', 'genaibench', 'simpleeval', 'sft_hack_datasets'}


@dataclass
class DataloaderResult:
    """
    Unified result from prepare_unified_dataloader.
    Provides consistent interface for map-style and iterable datasets.
    """
    loader: Union[DataLoader, wds.WebLoader]
    sampler: Optional[DistributedSampler]
    dataset_size: int
    is_iterable: bool = False
    _wds_pipeline: Optional[object] = field(default=None, repr=False)
    _batch_size: int = 1
    _num_workers: int = 4
    _world_size: int = 1
    virtual_epoch_steps: Optional[int] = None

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling. Works for both dataset types.

        For map-style: calls sampler.set_epoch()
        For WebDataset: recreates pipeline with new seed (uses virtual_epoch_steps if set)
        """
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        elif self._wds_pipeline is not None:
            self._recreate_wds_loader(epoch)

    def _recreate_wds_loader(self, epoch: int):
        """Recreate WebDataset loader for new epoch."""
        dataset = self._wds_pipeline.create_pipeline(epoch=epoch)
        steps = self.virtual_epoch_steps or (self.dataset_size // (self._batch_size * self._world_size))
        loader = wds.WebLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self.loader = loader.with_epoch(steps)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        if self.virtual_epoch_steps is not None:
            return self.virtual_epoch_steps
        if self.is_iterable:
            return self.dataset_size // (self._batch_size * self._world_size)
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)


class _T2IHFDataset(Dataset):
    """Internal HuggingFace dataset wrapper for MSCOCO/MJHQ T2I datasets."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "val",
        transform: Optional[transforms.Compose] = None,
        data_dir: Optional[str] = "./data",
    ):
        from datasets import load_from_disk

        # Load from local Arrow format (e.g., data/mscoco/val)
        local_path = Path(data_dir) / dataset_name / split
        self.hf_dataset = load_from_disk(str(local_path))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        sample = self.hf_dataset[idx]
        text = sample['text']

        if 'image' in sample:
            image = sample['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.empty(0)

        return image, text


def prepare_unified_dataloader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose] = None,
    condition_type: str = "text",
    shuffle: bool = True,
    virtual_epoch_steps: Optional[int] = None,
) -> DataloaderResult:
    """
    Unified dataloader factory for ImageNet, BLIP3O, MSCOCO, and MJHQ.

    Args:
        config: Dataset configuration dict with structure:
            {
                'target': 'imagenet' | 'blip3o' | 'mscoco' | 'mjhq' | 'geneval',
                'type': 'hf' | 'latents' | 'wds',
                'params': {
                    'data_dir': str,
                    'split': str,
                    'splits': ['journeydb', 'short-caption'],  # for BLIP3O
                    ...
                }
            }
        image_size: Target image resolution
        batch_size: Per-GPU batch size
        num_workers: Number of dataloader workers
        rank: Distributed training rank
        world_size: Total number of GPUs
        transform: Optional custom transform
        condition_type: 'label' or 'text'
        shuffle: Whether to shuffle data (False for eval)

    Returns:
        DataloaderResult with unified interface
    """
    target = config.get("target", "imagenet")

    if target in T2I_HF_DATASETS:
        result = _prepare_t2i_hf_loader(
            target, config, image_size, batch_size, num_workers, rank, world_size, transform, shuffle
        )
    elif target == "blip3o":
        result = _prepare_blip3o_loader(
            config, image_size, batch_size, num_workers, world_size, transform
        )
    elif target == "imagenet":
        result = _prepare_imagenet_loader(
            config, image_size, batch_size, num_workers, rank, world_size, transform, condition_type, shuffle
        )
    result.virtual_epoch_steps = virtual_epoch_steps
    return result


def _prepare_blip3o_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int,
    transform: Optional[transforms.Compose],
) -> DataloaderResult:
    """Prepare BLIP3O WebDataset loader."""
    from .blip3o_wds_dataset import BLIP3OWebDataset

    data_dir = config.get("data_dir", "./data/blip3o")
    # Support both 'splits' (list) and 'split' (single) keys
    splits = config.get("splits", config.get("split", "short-caption"))
    shuffle_buffer = config.get("shuffle_buffer", 10000)
    seed = config.get("seed", 42)

    wds_pipeline = BLIP3OWebDataset(
        data_dir=data_dir,
        splits=splits,
        transform=transform,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    dataset = wds_pipeline.create_pipeline(epoch=0)
    total_samples = wds_pipeline.estimated_size
    steps = total_samples // (batch_size * world_size)

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    # Bound epoch to exactly `steps` batches (with_epoch stops iteration, with_length only sets __len__)
    loader = loader.with_epoch(steps)

    return DataloaderResult(
        loader=loader,
        sampler=None,
        dataset_size=total_samples,
        is_iterable=True,
        _wds_pipeline=wds_pipeline,
        _batch_size=batch_size,
        _num_workers=num_workers,
        _world_size=world_size,
    )


def _prepare_t2i_hf_loader(
    dataset_name: str,
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    shuffle: bool,
) -> DataloaderResult:
    """Prepare MSCOCO/MJHQ HuggingFace loader."""
    split = config.get("split", "val")
    data_dir = config.get("data_dir", "./data")  # Local data directory

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    dataset = _T2IHFDataset(
        dataset_name=dataset_name,
        split=split,
        transform=transform,
        data_dir=data_dir,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # drop_last=True for train, False for eval
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    return DataloaderResult(
        loader=loader,
        sampler=sampler,
        dataset_size=len(dataset),
        is_iterable=False,
    )


def _prepare_imagenet_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    condition_type: str,
    shuffle: bool = True,
) -> DataloaderResult:
    """Prepare ImageNet-style loader using existing dataset classes."""
    data_dir = config.get("data_dir", "./data/imagenet")
    split = config.get("split", "train")

    # Build transform if not provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    # Create dataset based on type
    dataset = ImageNetHFDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        condition_type=condition_type,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # drop_last=True for train, False for eval
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    return DataloaderResult(
        loader=loader,
        sampler=sampler,
        dataset_size=len(dataset),
        is_iterable=False,
    )
