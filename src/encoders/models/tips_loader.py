import importlib
import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms


def make_tips_transform(resize_size: int = 224):
    """TIPS preprocessing: x/255 + resize. No ImageNet normalization."""
    to_tensor = transforms.Lambda(lambda x: x / 255.)
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    return transforms.Compose([to_tensor, resize])


# model_config -> (factory_fn_name, ffn_layer)
MODEL_CONFIGS = {
    's14':  ('vit_small',  'mlp'),
    'b14':  ('vit_base',   'mlp'),
    'l14':  ('vit_large',  'mlp'),
    'so14': ('vit_so400m', 'mlp'),
    'g14':  ('vit_giant2', 'swiglu'),
}

HF_MODEL_MAP = {
    'b14': 'google/tipsv2-b14',
    'l14': 'google/tipsv2-l14',
    'so14': 'google/tipsv2-so400m14',
    'g14': 'google/tipsv2-g14',
}

TIPSV1_CHECKPOINT_FILENAMES = {
    's14':   'tips_oss_s14_highres_distilled_vision.npz',
    'b14':   'tips_oss_b14_highres_distilled_vision.npz',
    'l14':   'tips_oss_l14_highres_distilled_vision.npz',
    'so14':  'tips_oss_so400m14_highres_largetext_distilled_vision.npz',
    'g14hr': 'tips_oss_g14_highres_vision.npz',
    'g14lr': 'tips_oss_g14_lowres_vision.npz',
}


_ie_module = None

def _get_image_encoder_module(source_dir):
    """Import image_encoder from directory via sys.path (torch.compile compatible)."""
    global _ie_module
    if _ie_module is not None:
        return _ie_module
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    _ie_module = importlib.import_module('image_encoder')
    return _ie_module


def _create_model(ie_module, model_config, img_size):
    """Create TIPS vision encoder from factory function."""
    arch_config = 'g14' if model_config in ('g14hr', 'g14lr') else model_config
    factory_name, ffn_layer = MODEL_CONFIGS[arch_config]
    factory_fn = getattr(ie_module, factory_name)
    return factory_fn(
        img_size=img_size,
        patch_size=14,
        ffn_layer=ffn_layer,
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )


def load_tipsv2(model_config):
    """Load TIPSv2 vision encoder from HuggingFace.

    Downloads image_encoder.py + safetensors directly (no trust_remote_code),
    so torch.compile works.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    hf_name = HF_MODEL_MAP[model_config]

    ie_path = hf_hub_download(hf_name, "image_encoder.py")
    config_path = hf_hub_download(hf_name, "config.json")
    weights_path = hf_hub_download(hf_name, "model.safetensors")

    ie = _get_image_encoder_module(os.path.dirname(ie_path))

    with open(config_path) as f:
        config = json.load(f)

    model = _create_model(ie, model_config, img_size=config['img_size'])

    # Load only vision encoder weights from safetensors
    all_weights = load_file(weights_path)
    vision_weights = {k.replace('vision_encoder.', ''): v
                      for k, v in all_weights.items()
                      if k.startswith('vision_encoder.')}
    model.load_state_dict(vision_weights)
    return model


def load_tipsv1(model_config, img_size=448):
    """Load TIPSv1 vision encoder from local repo + NPZ checkpoint.

    Requires TIPS_REPO_DIR and TIPS_CKPT_DIR environment variables.
    """
    repo_dir = os.environ.get("TIPS_REPO_DIR")
    ckpt_dir = os.environ.get("TIPS_CKPT_DIR")
    if repo_dir is None or ckpt_dir is None:
        raise ValueError("TIPS_REPO_DIR and TIPS_CKPT_DIR must be set")

    ie = _get_image_encoder_module(os.path.join(repo_dir, 'pytorch'))
    model = _create_model(ie, model_config, img_size)

    ckpt_path = os.path.join(ckpt_dir, TIPSV1_CHECKPOINT_FILENAMES[model_config])
    checkpoint = dict(np.load(ckpt_path, allow_pickle=False))
    state_dict = {k: torch.tensor(v) for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    return model
