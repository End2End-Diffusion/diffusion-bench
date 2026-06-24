"""
Stage 2 (diffusion/flow matching) config dataclasses.

Contains:
- TransportConfig: Flow matching transport settings
- SamplerConfig: ODE/SDE sampler settings
- GuidanceConfig: Eval-time guidance settings (CFG and/or IG)
- RepaConfig: REPA loss settings
- ConditioningConfig: Conditioning settings (label vs text)
- Stage2Config: Top-level config for Stage 2 training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from .shared import DatasetConfig, EvalConfig, MiscConfig, ModelConfig, TrainingConfig


@dataclass
class MeanflowConfig:
    """Meanflow configuration for flow matching."""
    fm_ratio: float = 0.75
    norm_p: float = 1.0
    norm_eps: float = 0.01
    # MeanFlow-style CFG distillation, baked into the target velocity (omega == 1.0 disables it).
    cfg_omega: float = 1.0      # guidance scale
    cfg_kappa: float = 0.5      # secondary mix weight
    cfg_t_start: float = 0.0    # guidance active only for t in [cfg_t_start, cfg_t_end]
    cfg_t_end: float = 1.0


@dataclass
class TransportConfig:
    """Transport configuration for flow matching."""
    prediction: str = "velocity"  # "velocity" or "x"
    time_dist_type: str = "logit-normal_0_1"
    t_eps: float = 0.05
    percep_loss_t_thresh: float = 0.7
    meanflow: Optional[MeanflowConfig] = None


@dataclass
class PerceptualLossConfig:
    encoders: str = ""
    percep_loss_weights: List[float] = field(default_factory=list)


@dataclass
class SamplerConfig:
    """Sampler configuration for ODE Euler flow matching."""
    num_steps: int = 50


@dataclass
class CFGConfig:
    """CFG configuration for test-time guidance."""
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


@dataclass
class IGConfig:
    """IG configuration for test-time guidance."""
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    unconditional_scale: Optional[float] = None


@dataclass
class GuidanceConfig:
    """Guidance configuration for test-time guidance."""
    cfg: Optional[CFGConfig] = field(default_factory=CFGConfig)
    ig: Optional[IGConfig] = field(default_factory=IGConfig)
    disabled: bool = False

    @property
    def use_cfg(self):
        return not self.disabled and self.cfg is not None and self.cfg.scale > 1.0

    @property
    def use_ig(self):
        return not self.disabled and \
            self.ig is not None and \
            (self.ig.scale > 1.0 or \
            self.ig.unconditional_scale is not None and \
            self.ig.unconditional_scale != 1.0)

    @property
    def any_guidance_active(self):
        return self.use_cfg or self.use_ig

    def get_mode_string(self):
        if self.use_ig and self.use_cfg:
            return "ig+cfg"
        if self.use_ig:
            return "ig"
        if self.use_cfg:
            return "cfg"
        return "none"


@dataclass
class RepaConfig:
    """REPA loss configuration with multi-layer support."""
    use_repa: bool = False
    use_reg: bool = False
    reg_coeff: float = 0.03
    repa_layer_depth: int = 8
    repa_coeff: float = 0.5
    target_encoder: str = "dinov2-vit-b"
    target_encoder_resolution: int = 256
    z_dim: Optional[int] = None  # initialized later in train.py


@dataclass
class ConditioningArchConfig:
    """In-context conditioning architecture configuration."""
    num_t_tokens: int = 4
    num_c_tokens: int = 8


@dataclass
class TextEncoderConfig:
    """Text encoder configuration."""
    model_name: str = "Qwen/Qwen3-0.6B"
    max_length: int = 256


@dataclass
class ConditioningConfig:
    """Conditioning configuration for ImageNet and T2I"""
    type: str = "label"
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    cfg_dropout_prob: float = 0.1
    context_dim: Optional[int] = None  # initialized later in train.py
    arch: ConditioningArchConfig = field(default_factory=ConditioningArchConfig)


@dataclass
class InternalGuidanceConfig:
    """Internal guidance training-related configuration."""
    base_model_depth: Optional[int] = None
    base_model_coeff: float = 1.0


@dataclass
class Stage2Config:
    """Top-level configuration for Stage 2 training."""
    stage_1: ModelConfig = field(default_factory=ModelConfig)
    stage_2: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    repa: RepaConfig = field(default_factory=RepaConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
    internal_guidance: InternalGuidanceConfig = field(default_factory=InternalGuidanceConfig)
    perceptual_loss: PerceptualLossConfig = field(default_factory=PerceptualLossConfig)
    eval: Optional[EvalConfig] = None

    def post_process(self):
        """Post-process the config to set certain runtime fields."""
        if self.transport.meanflow is not None:
            self.guidance.disabled = True

        if self.conditioning.type == "label" and self.dataset.condition_type is not None:
            self.conditioning.type = self.dataset.condition_type

        if self.conditioning.type == "text":
            self.conditioning.arch.num_c_tokens = self.conditioning.text_encoder.max_length

    def prepare_model_params(self):
        """Populate stage_2.params from typed config fields for model construction.

        Call this after setting runtime fields (conditioning.text_feature_dim,
        conditioning.context_dim, repa.z_dim) and before instantiating the model.
        Uses setdefault for static values so YAML-specified params are never overwritten.
        """
        params = self.stage_2.params

        # Conditioning
        params.setdefault('condition_type', self.conditioning.type)
        params.setdefault('num_classes', self.misc.num_classes)
        params.setdefault('context_dim', self.conditioning.context_dim)

        if self.repa.use_reg:
            params.setdefault('enable_reg', True)

        # REPA
        if self.repa.use_repa:
            params.setdefault('enable_repa', True)
            params.setdefault('repa_layer_depth', self.repa.repa_layer_depth)

        if (self.repa.use_reg or self.repa.use_repa) and self.repa.z_dim is not None:
            params.setdefault('z_dim', self.repa.z_dim)

        # Conditioning architecture
        params.setdefault('cond_arch', self.conditioning.arch)

        # MeanFlow conditions on two times (t, t-r) via a second time embedder
        if self.transport.meanflow is not None:
            params.setdefault('is_meanflow', True)

        # Internal guidance
        if self.internal_guidance.base_model_depth is not None:
            params.setdefault('base_model_depth', self.internal_guidance.base_model_depth)
