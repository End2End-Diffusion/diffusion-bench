import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class RoPE(nn.Module):
    def __init__(self, dim, vis_len, cond_len=0, extra_tokens=0, theta=10000.,):
        super().__init__()
        # 2D RoPE for vision
        d, T = dim // 2, int(vis_len ** 0.5)
        vis_freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))  # [D//4]
        vis_base_angles = torch.outer(torch.arange(T).float(), vis_freqs)  # [T, D//4]
        vis_angles = torch.cat([
            vis_base_angles[:, None].expand(-1, T, -1),
            vis_base_angles[None, :].expand(T, -1, -1)
        ], dim=-1).reshape(vis_len, d)  # [T, T, D//2] -> [L', D//2]
        # no PE for extra (cls) or cond tokens
        extra_angles = torch.zeros(extra_tokens, dim // 2)
        cond_angles = torch.zeros(cond_len, dim // 2)
        angles = torch.cat([extra_angles, vis_angles, cond_angles], dim=0).repeat_interleave(2, dim=-1)  # [L, D]
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class NormAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads, self.dim, self.head_dim = num_heads, dim, dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope, attn_mask=None):
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        return self.proj(out)


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size, n_tokens=4, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # n_tokens=0: no time tokens, forward returns only the pooled embedding [B, H] (e.g. for adaLN)
        self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size))) if n_tokens else None

    def forward(self, t, return_base_embed=False):
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        if self.learnable_tokens is None:
            return t_embed
        if return_base_embed:
            t_embed = t_embed.unsqueeze(1)
            return t_embed, self.learnable_tokens + t_embed
        else:
            return self.learnable_tokens + t_embed.unsqueeze(1)


class RefinerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = NormAttention(dim, dim // 128)
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))
        self.rope = nn.Identity()  # no positional encoding inside the refiner
        # zero-init residual branches: each block starts as identity
        for layer in (self.attn.proj, self.mlp.w3):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=self.rope, attn_mask=attn_mask)
        return x + self.mlp(self.norm2(x))


class TextRefiner(nn.Module):
    """[B, T, L, D] text features -> attention across layers -> learned layer collapse
    -> attention across tokens -> [B, T, H]. Starts as mean over layers + projection."""
    def __init__(self, context_dim, hidden_size, n_tokens, num_text_layers, num_layerwise_blocks=2, num_refiner_blocks=2):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([RefinerBlock(context_dim) for _ in range(num_layerwise_blocks)])
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        nn.init.constant_(self.projector.weight, 1.0 / num_text_layers)
        self.norm = RMSNorm(context_dim)
        self.proj = nn.Linear(context_dim, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, hidden_size))
        self.refiner_blocks = nn.ModuleList([RefinerBlock(hidden_size) for _ in range(num_refiner_blocks)])

    def forward(self, y, attn_mask=None):
        B, T, L, D = y.shape
        x = y.reshape(B * T, L, D)
        for blk in self.layerwise_blocks:
            x = blk(x)
        x = self.projector(x.reshape(B, T, L, D).transpose(2, 3)).squeeze(-1)
        x = self.proj(self.norm(x)) + self.pos_emb[:, :T]
        if attn_mask is not None:
            attn_mask = (1.0 - attn_mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
        for blk in self.refiner_blocks:
            x = blk(x, attn_mask)
        return x


class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_size, num_classes=1000, context_dim=768, condition_type="label", n_tokens=8, refiner_cfg=None):
        super().__init__()
        self.condition_type = condition_type
        self.hidden_size = hidden_size

        if condition_type == "label":
            self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
            self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))
        elif condition_type == "text" and refiner_cfg is not None:
            self.refiner = TextRefiner(context_dim, hidden_size, n_tokens, refiner_cfg.num_text_layers,
                                       refiner_cfg.num_layerwise_blocks, refiner_cfg.num_refiner_blocks)
        elif condition_type == "text":
            self.norm = RMSNorm(context_dim)
            self.proj = nn.Linear(context_dim, hidden_size)
        else:
            raise ValueError(f"Unknown condition_type: {condition_type}")

    def forward(self, y, attn_mask=None):
        if self.condition_type == "label":
            return self.learnable_tokens + self.embedding_table(y).unsqueeze(1)
        if hasattr(self, "refiner"):
            return self.refiner(y, attn_mask)
        return self.proj(self.norm(y))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
