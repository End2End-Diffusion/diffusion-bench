import os

import torch
from torchvision import transforms


def make_eupe_transform(resize_size: int = 224):
    to_tensor = transforms.Lambda(lambda x: x / 255.)
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


MODEL_NAMES = {
    'eupe_vitt16',
    'eupe_vits16',
    'eupe_vitb16',
}
CHECKPOINT_FILENAMES = {
    'eupe_vitt16': 'EUPE-ViT-T.pt',
    'eupe_vits16': 'EUPE-ViT-S.pt',
    'eupe_vitb16': 'EUPE-ViT-B.pt',
}
def load_eupe(model_name):
    assert model_name in MODEL_NAMES
    repo_dir = os.environ.get("EUPE_REPO_DIR")
    ckpt_dir = os.environ.get("EUPE_CKPT_DIR")
    if repo_dir is None or ckpt_dir is None:
        raise ValueError("EUPE_REPO_DIR and EUPE_CKPT_DIR must be set as environment variables")
    model = torch.hub.load(
        repo_dir,
        model_name,
        source='local',
        weights=os.path.join(ckpt_dir, CHECKPOINT_FILENAMES[model_name])
    )
    return model
