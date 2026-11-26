import torch
import torch.nn as nn

from .lightningDiT import LightningDiT, LightningDiTBlock, LightningFinalLayer
from .model_utils import RoPE, modulate


class MMDiTSingleBlock(LightningDiTBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio)
        # adaLN-single: per-block shift/scale/gate bias on top of the shared timestep modulation
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, mod, rope, attn_mask=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = (self.scale_shift_table + mod.unflatten(1, (6, -1))).unbind(1)
        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1), rope=rope, attn_mask=attn_mask)
        return x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))


class MMDiTSingleFinalLayer(LightningFinalLayer):
    def __init__(self, hidden_size, patch_size, out_channels, cls_dim=None):
        super().__init__(hidden_size, patch_size, out_channels, cls_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, mod):
        shift, scale = (self.scale_shift_table + mod.unflatten(1, (2, -1))).unbind(1)
        x = modulate(self.norm_final(x), shift, scale)
        if hasattr(self, 'cls_linear'):
            return self.linear(x[:, 1:, :]), self.cls_linear(x[:, 0, :])
        return self.linear(x)


class MMDiTSingle(LightningDiT):
    """Single-stream MMDiT: image + text tokens in one sequence with shared weights,
    timestep injected through adaLN-single (no time tokens)."""
    def __init__(self, depth=28, num_heads=16, mlp_ratio=4.0, enable_reg=False, z_dim=None,
                 cond_arch=None, is_meanflow=False, **kwargs):
        if is_meanflow:
            raise NotImplementedError("MMDiTSingle does not support MeanFlow.")
        super().__init__(depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, enable_reg=enable_reg,
                         z_dim=z_dim, cond_arch=cond_arch, **kwargs)
        h = self.hidden_size
        self.blocks = nn.ModuleList([MMDiTSingleBlock(h, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = MMDiTSingleFinalLayer(h, self.patch_size, self.in_channels, cls_dim=z_dim if enable_reg else None)
        self.modulation_embedder = nn.Sequential(nn.SiLU(), nn.Linear(h, 6 * h))
        self.final_modulation_embedder = nn.Sequential(nn.SiLU(), nn.Linear(h, 2 * h))
        self.num_cond_tokens = cond_arch.num_c_tokens
        self.rope = RoPE(h // num_heads, self.x_embedder.num_patches, self.num_cond_tokens, extra_tokens=int(enable_reg))
        # only the pooled time embedding feeds adaLN; keep the unused tokens as a buffer (same state-dict key)
        # so DDP doesn't see a parameter without gradients (train.py enables grads on all parameters)
        tokens = self.t_embedder.learnable_tokens.detach()
        del self.t_embedder.learnable_tokens
        self.t_embedder.register_buffer("learnable_tokens", tokens)

        self.initialize_weights()
        for layer in (self.modulation_embedder[1], self.final_modulation_embedder[1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _build_sequence(self, x, t, condition_kwargs):
        seq = [self.x_embedder(x), self.ctx_embedder(condition_kwargs["context"], condition_kwargs.get("attn_mask"))]
        if self.enable_reg:
            seq.insert(0, self.cls_in_norm(self.cls_in_proj(condition_kwargs["cls_t"])).unsqueeze(1))
        return torch.cat(seq, dim=1)

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        x = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(x, condition_kwargs)
        t_emb = self.t_embedder(t, return_base_embed=True)[0].squeeze(1)
        mod, final_mod = self.modulation_embedder(t_emb), self.final_modulation_embedder(t_emb)
        s, n = int(self.enable_reg), self.x_embedder.num_patches
        for i, block in enumerate(self.blocks):
            x = block(x, mod, self.rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(x[:, :s + n, :])

        out = self.final_layer(x[:, :s + n, :], final_mod)
        x = (self.unpatchify(out[0]), out[1]) if self.enable_reg else self.unpatchify(out)
        if return_intermediate:
            return x, zt_intermediate
        return x
