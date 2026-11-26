import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .model_utils import ConditionEmbedder, DiffusionTransformerBlock, GaussianFourierEmbedding, RMSNorm, RoPE, check_time_cond


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DDTDecoderBlock(DiffusionTransformerBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6*hidden_size)
        )

    def forward(self, x, c, rope, attn_mask=None):
        modulation = self.adaln_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope, attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cls_dim=None):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        if cls_dim is not None:
            self.cls_linear = nn.Linear(hidden_size, cls_dim)

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        if hasattr(self, 'cls_linear'):
            cls_pred = self.cls_linear(x[:, 0, :])
            return self.linear(x[:, 1:, :]), cls_pred
        return self.linear(x)


class DiTwDDTHead(nn.Module):
    def __init__(
        self,
        input_size=16,
        in_channels=768,
        patch_size=[1, 1],
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        enable_repa=False,
        repa_layer_depth=8,
        z_dim=None,
        enable_reg=False,
        num_classes=1000,
        condition_type="label",
        context_dim=768,
        cond_arch=None,
        is_meanflow=False,
        refiner_cfg=None,
        time_cond="in_context",  # encoder time: "in_context" tokens or "adaln" (adaLN-single); decoder is adaLN on s
    ):
        super().__init__()
        self.adaln = check_time_cond(time_cond, is_meanflow)
        self.in_channels = in_channels
        self.enc_hidden_size, dec_hidden_size = hidden_size
        self.num_enc_blocks, self.num_dec_blocks = depth
        self.s_patch_size, self.x_patch_size = patch_size
        enc_num_heads, dec_num_heads = num_heads

        self.repa_layer_depth = repa_layer_depth
        self.enable_reg = enable_reg
        self.is_meanflow = is_meanflow

        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, in_channels, self.enc_hidden_size)
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, in_channels, dec_hidden_size)
        self.s_projector = nn.Linear(self.enc_hidden_size, dec_hidden_size) if self.enc_hidden_size != dec_hidden_size else nn.Identity()

        # MeanFlow conditions on two times (t, t-r); add a second time embedder + tokens for absolute t
        n_t = 0 if self.adaln else cond_arch.num_t_tokens
        self.num_cond_tokens = n_t * (2 if is_meanflow else 1) + cond_arch.num_c_tokens
        self.t_embedder = GaussianFourierEmbedding(self.enc_hidden_size, n_t)
        if is_meanflow:
            self.t_abs_embedder = GaussianFourierEmbedding(self.enc_hidden_size, n_t)
        self.ctx_embedder = ConditionEmbedder(
            self.enc_hidden_size, num_classes, context_dim, condition_type, cond_arch.num_c_tokens, refiner_cfg
        )

        self.blocks = []
        for _ in range(self.num_enc_blocks):
            self.blocks.append(DiffusionTransformerBlock(self.enc_hidden_size, enc_num_heads, mlp_ratio, adaln=self.adaln))
        for _ in range(self.num_dec_blocks):
            self.blocks.append(DDTDecoderBlock(dec_hidden_size, dec_num_heads, mlp_ratio))
        self.blocks = nn.ModuleList(self.blocks)
        if self.adaln:  # one shared projection of the timestep embedding for all encoder blocks
            self.modulation_embedder = nn.Sequential(nn.SiLU(), nn.Linear(self.enc_hidden_size, 6 * self.enc_hidden_size))

        self.final_layer = DDTFinalLayer(dec_hidden_size, self.x_patch_size, in_channels, cls_dim=z_dim if enable_reg else None)
        self.enc_rope = RoPE(self.enc_hidden_size // enc_num_heads, self.s_embedder.num_patches, self.num_cond_tokens, extra_tokens=int(enable_reg))
        self.dec_rope = RoPE(dec_hidden_size // dec_num_heads, self.x_embedder.num_patches, extra_tokens=int(enable_reg))
        if enable_repa:
            self.repa_projector = nn.Linear(self.enc_hidden_size, z_dim)
        if enable_reg:
            self.cls_in_proj_enc = nn.Linear(z_dim, self.enc_hidden_size)
            self.cls_in_norm_enc = RMSNorm(self.enc_hidden_size)
            self.cls_in_proj_dec = nn.Linear(z_dim, dec_hidden_size)
            self.cls_in_norm_dec = RMSNorm(dec_hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Patch embedders
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Condition embedders
        if hasattr(self.ctx_embedder, "mlp"):
            nn.init.normal_(self.ctx_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ctx_embedder.mlp[2].weight, std=0.02)
        if hasattr(self.ctx_embedder, "embedding_table"):
            nn.init.normal_(self.ctx_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers
        if hasattr(self, "modulation_embedder"):
            nn.init.constant_(self.modulation_embedder[1].weight, 0)
            nn.init.constant_(self.modulation_embedder[1].bias, 0)
        for block in self.blocks:
            if hasattr(block, "adaln_modulation"):
                nn.init.constant_(block.adaln_modulation[-1].weight, 0)
                nn.init.constant_(block.adaln_modulation[-1].bias, 0)

        # Timestep embedding MLP
        for t_embedder in ("t_embedder", "t_abs_embedder"):
            if hasattr(self, t_embedder):
                nn.init.normal_(getattr(self, t_embedder).mlp[0].weight, std=0.02)
                nn.init.normal_(getattr(self, t_embedder).mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if hasattr(self.final_layer, "cls_linear"):
            nn.init.constant_(self.final_layer.cls_linear.weight, 0)
            nn.init.constant_(self.final_layer.cls_linear.bias, 0)

    def unpatchify(self, x, p):
        """[N, T, patch_size**2 * C] -> [N, C, H, W]"""
        h, c = int(x.shape[1] ** 0.5), self.in_channels
        x = x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h*p, h*p)
        return x

    def _build_sequence(self, x, t, condition_kwargs):
        """Returns sequence concatenated with all condition tokens, and the base timestep embedding (no learnable tokens)"""
        seq = []
        if self.enable_reg:
            cls_in = self.cls_in_norm_enc(self.cls_in_proj_enc(condition_kwargs["cls_t"]))
            seq.append(cls_in.unsqueeze(1))
        seq.append(self.s_embedder(x))
        if self.adaln:
            t_emb_base = self.t_embedder(t).unsqueeze(1)
        else:
            t_emb_base, t_emb = self.t_embedder(t, return_base_embed=True)
            seq.append(t_emb)
            if self.is_meanflow:
                seq.append(self.t_abs_embedder(condition_kwargs["t_abs"]))
        seq.append(self.ctx_embedder(condition_kwargs["context"], condition_kwargs.get("attn_mask")))
        seq = torch.cat(seq, dim=1)
        return seq, t_emb_base

    def _build_attn_mask(self, seq, condition_kwargs):
        # Create multiplicative mask template
        attn_mask = torch.ones((seq.shape[0], seq.shape[1]), device=seq.device)
        cond_mask = condition_kwargs.get("attn_mask")
        if cond_mask is not None:
            attn_mask[:, -cond_mask.shape[1]:] = cond_mask
        # Convert to additive mask
        attn_mask = (1.0 - attn_mask[:, None, None, :]) * torch.finfo(seq.dtype).min
        return attn_mask

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        mod = self.modulation_embedder(t_emb_base.squeeze(1)) if self.adaln else None
        s, n = int(self.enable_reg), self.s_embedder.num_patches
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask, mod)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, :s + n, :])
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :s + n, :]))

        x = self.x_embedder(x)
        if self.enable_reg:
            cls_in = self.cls_in_norm_dec(self.cls_in_proj_dec(condition_kwargs["cls_t"]))
            x = torch.cat([cls_in.unsqueeze(1), x], dim=1)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        if self.enable_reg:
            x, cls_pred = self.final_layer(x, seq)
        else:
            x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)
        if self.enable_reg:
            x = (x, cls_pred)

        if return_intermediate:
            return x, zt_intermediate
        return x


class DiTwDDTHeadIG(DiTwDDTHead):
    def __init__(self, base_model_depth=8, **kwargs):
        super().__init__(**kwargs)
        self.base_model_depth = base_model_depth

        self.base_final_layer = DDTFinalLayer(self.enc_hidden_size, self.s_patch_size, self.in_channels)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.base_final_layer.linear.weight, 0)
        nn.init.constant_(self.base_final_layer.linear.bias, 0)

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        x_base = None
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        mod = self.modulation_embedder(t_emb_base.squeeze(1)) if self.adaln else None
        s, n = int(self.enable_reg), self.s_embedder.num_patches
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask, mod)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, :s + n, :])
            if (i + 1) == self.base_model_depth:
                x_base = seq[:, s:s + n, :]
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :s + n, :]))

        x = self.x_embedder(x)
        if self.enable_reg:
            cls_in = self.cls_in_norm_dec(self.cls_in_proj_dec(condition_kwargs["cls_t"]))
            x = torch.cat([cls_in.unsqueeze(1), x], dim=1)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        if self.enable_reg:
            x, cls_pred = self.final_layer(x, seq)
        else:
            x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)

        x_base = F.silu(t_emb_base + x_base)
        x_base = self.base_final_layer(x_base, x_base)
        x_base = self.unpatchify(x_base, self.s_patch_size)

        out = (x, x_base, cls_pred) if self.enable_reg else (x, x_base)
        if return_intermediate:
            return out, zt_intermediate
        return out
