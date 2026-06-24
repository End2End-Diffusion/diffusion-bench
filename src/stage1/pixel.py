import torch.nn as nn
import torch

class PixelEncoder(nn.Module):
    def __init__(
        self,
        resolution: int = 256,
    ):
        super().__init__()
        self.resolution = resolution

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images for PixelEncoder

        Args:
            x: Images in [0, 1] range, shape (B, 3, H, W)

        Returns:
            Images in [-1, 1] range, resized to target resolution
        """
        # Resize if needed
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(
                x,
                size=(self.resolution, self.resolution),
                mode='bilinear',
                align_corners=False
            )

        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent representations.

        Args:
            x: Images in [0, 1] range, shape (B, 3, H, W)

        Returns:
            Latent representations, shape (B, C, H, W)
        """
        return self._preprocess(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to images.

        Args:
            z: Latent representations, shape (B, C, H, W)

        Returns:
            Images in [0, 1] range, shape (B, 3, H, W)
        """
        return (z + 1.0) / 2.0
