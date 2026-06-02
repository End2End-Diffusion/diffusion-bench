"""
Encoder factory for seamless encoder switching.

Provides a simple string-based interface for creating encoders, using the
unified vision_encoder system.

This module serves as a compatibility layer between RAE and the vision_encoder system.
"""

import torch

from .vision_encoder import create_encoder as _vision_encoder_create


def create_encoder_from_string(encoder_name: str, device: torch.device = None,
                               resolution: int = 256, accelerator=None):
    """
    Factory function to create encoder from string name.

    This is a wrapper that uses the vision_encoder.create_encoder() function
    with encoder names in the format: encoder_type-architecture-model_config

    Args:
        encoder_name: Encoder name string (e.g., 'dinov2-vit-b', 'dinov3-vit-b16')
        device: torch.device (optional, defaults to cuda if available)
        resolution: RAE operating resolution (default: 256)
        accelerator: Optional accelerator for distributed training

    Returns:
        VisionEncoder instance (handles its own preprocessing/normalization)

    Example:
        >>> encoder = create_encoder_from_string('dinov2-vit-b')
        >>> encoder = create_encoder_from_string('dinov3-vit-l16', resolution=512)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create encoder using vision_encoder system
    encoder = _vision_encoder_create(
        encoder_name,
        device=device,
        resolution=resolution,
        accelerator=accelerator
    )

    # Ensure patch_size is set for RAE compatibility
    if encoder.patch_size is None:
        # Infer patch size from model config or set default
        if hasattr(encoder, 'model') and hasattr(encoder.model, 'patch_size'):
            encoder.patch_size = encoder.model.patch_size
        elif '16' in encoder_name:
            encoder.patch_size = 16
        elif '14' in encoder_name:
            encoder.patch_size = 14
        else:
            encoder.patch_size = 16  # Default

    return encoder
