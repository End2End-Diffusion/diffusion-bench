from encoders.vision_encoder import load_encoders
from stage1.disc.lpips import LPIPS
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self, encoder_type, percep_loss_weights=None, resolution=256, device="cuda"):
        super().__init__()
        self.encoders = []
        self.lpips = None
        self.lpips_idx = None
        self.percep_loss_weights = []

        if not encoder_type:
            return

        enc_names = [s.strip() for s in encoder_type.split(',')]
        # Positional weights aligned with enc_names (i.e. with the final
        # forward() output order, which keeps lpips at its original index).
        self.percep_loss_weights = (
            list(percep_loss_weights) if percep_loss_weights is not None
            else [1.0] * len(enc_names)
        )

        if 'lpips' in enc_names:
            self.lpips_idx = enc_names.index('lpips')
            enc_names.remove('lpips')
            self.lpips = LPIPS().eval().to(device)
        encoder_str = ','.join(enc_names)
        self.encoders = load_encoders(encoder_str, device=device, resolution=resolution) if enc_names else []

        # Turn off gradient computation for encoders
        for encoder in self.encoders:
            encoder.requires_grad_(False)
        if self.lpips is not None:
            self.lpips.requires_grad_(False)

    def forward(self, x_pred, x_gt):
        # Inputs are in [-1, 1]
        losses = []
        for encoder in self.encoders:
            # Vision Encoders expect [0, 255]
            x_pred_ = encoder.preprocess((x_pred + 1.0) * 127.5)
            x_gt_ = encoder.preprocess((x_gt + 1.0) * 127.5)
            feats_x_pred = encoder.forward_features(x_pred_)['x_norm_patchtokens']  # [B, T, D]
            feats_x_gt = encoder.forward_features(x_gt_)['x_norm_patchtokens']  # [B, T, D]
            losses.append(((feats_x_pred - feats_x_gt) ** 2).mean(dim=(1, 2)))  # [B, T, D] -> [B]

        if self.lpips_idx is not None:
            # LPIPS requires [-1, 1]
            lpips_loss = self.lpips(x_pred, x_gt, reduction="none").mean(dim=(1, 2, 3))  # [B, C, H, W] -> [B]
            losses.insert(self.lpips_idx, lpips_loss)
        # self.percep_loss_weights is aligned with the original comma-split order,
        # which matches `losses` after the insert above.
        return sum(w * L for w, L in zip(self.percep_loss_weights, losses))
