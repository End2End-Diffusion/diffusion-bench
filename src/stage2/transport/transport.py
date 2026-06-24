import random

import torch as th
import torch.nn.functional as F

from stage2.utils import apply_cfg_dropout
from utils.dist_utils import synchronize_gradients


def _expand_t(t, x):
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


def get_time_sampler(time_dist_type: str):
    parts = time_dist_type.split("_")
    name = parts[0]
    if name == "logit-normal":
        assert len(parts) == 3, f"Expected 'logit-normal_MU_SIGMA', got '{time_dist_type}'"
        mu, sigma = float(parts[1]), float(parts[2])
        assert sigma > 0, "sigma must be > 0"
        return lambda bs: (th.randn(bs) * sigma + mu).sigmoid()
    else:
        raise NotImplementedError(f"Unknown time distribution: {time_dist_type}")


class Transport:
    def __init__(self, prediction="velocity", time_dist_type="logit-normal_0_1", time_dist_shift=1.0, time_dist_shift_eval=1.0, t_eps=0.05, percep_loss_t_thresh=0.7):
        self.prediction = prediction
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.time_dist_shift_eval = time_dist_shift_eval
        self.t_eps = t_eps
        self.percep_loss_t_thresh = percep_loss_t_thresh
        self.time_sampler = get_time_sampler(time_dist_type)

    def sample(self, x1):
        x0 = th.randn_like(x1)
        t = self.time_sampler(x1.shape[0]).to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################
    def training_losses(self, model, x1, model_kwargs={}, model_kwargs_null={}, z_clean=None, repa_coeff=None, base_model_coeff=1.0, percep_loss=None, cfg_dropout_prob=0.1, ema_model=None, cls_clean=None, reg_coeff=None):
        model_kwargs, _ = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)

        t, x0, x1 = self.sample(x1)
        xt = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        vt = (xt - x1) / _expand_t(t, xt).clamp_min(self.t_eps)

        enable_repa = z_clean is not None and repa_coeff is not None
        enable_reg = cls_clean is not None and reg_coeff is not None
        # Append cls noising using the same t and formula
        if enable_reg:
            cls_x0 = th.randn_like(cls_clean)
            cls_t = (1 - _expand_t(t, cls_clean)) * cls_clean + _expand_t(t, cls_clean) * cls_x0
            v_cls = (cls_t - cls_clean) / _expand_t(t, cls_t).clamp_min(self.t_eps)
            model_kwargs = {**model_kwargs, "cls_t": cls_t}

        zt_pred = None
        if enable_repa:
            model_output, zt_pred = model(xt, t, return_intermediate=True, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)

        # Handle multi-output models: REG (full, cls), IG (full, base), or REG+IG (full, base, cls)
        base_output = None
        cls_pred = None
        if isinstance(model_output, tuple):
            if len(model_output) == 3:
                model_output, base_output, cls_pred = model_output
            elif len(model_output) == 2:
                if enable_reg:
                    model_output, cls_pred = model_output
                else:
                    model_output, base_output = model_output

        # Compute loss
        terms = {'loss': self.compute_loss(model_output, vt, xt, t)}
        if base_output is not None:
            loss_base = self.compute_loss(base_output, vt, xt, t)
            terms['loss'] = terms['loss'] + base_model_coeff * loss_base
            terms['loss_base'] = loss_base
        loss_repa = th.tensor(0.0, device=x1.device)
        if enable_repa and zt_pred is not None:
            loss_repa = repa_coeff * F.mse_loss(zt_pred, z_clean)
        terms['loss_repa'] = loss_repa
        loss_reg = th.tensor(0.0, device=x1.device)
        if enable_reg and cls_pred is not None:
            loss_reg = reg_coeff * F.mse_loss(self.convert_model_pred(cls_pred, cls_t, t), v_cls)
        terms['loss_reg'] = loss_reg

        if percep_loss is not None:
            assert self.prediction == "x"
            # Mask based on t < percep_loss_t_thresh
            mask = t < self.percep_loss_t_thresh
            terms['loss_percep'] = percep_loss(model_output, x1) * mask  # [B]

        return terms

    def post_backward(self, model):
        pass

    def convert_model_pred(self, output, xt, t):
        # Unify model output to v-pred
        if self.prediction == "velocity":
            return output
        elif self.prediction == "x":
            t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
            return (xt - output) / t_safe

    def compute_loss(self, output, vt, xt, t):
        output = self.convert_model_pred(output, xt, t)
        return (output - vt) ** 2

    def get_drift(self):
        def body_fn(x, t, h, model, **model_kwargs):
            cls_t = model_kwargs.get('cls_t')
            if cls_t is not None:
                x_pred, cls_pred = model(x, t, **model_kwargs)
                return self.convert_model_pred(x_pred, x, t), self.convert_model_pred(cls_pred, cls_t, t)
            model_output = model(x, t, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            return self.convert_model_pred(model_output, x, t)
        return body_fn


class TransportMF(Transport):
    def __init__(self, fm_ratio=0.75, norm_p=1.0, norm_eps=0.01,
                 cfg_omega=1.0, cfg_kappa=0.5, cfg_t_start=0.0, cfg_t_end=1.0, **kwargs):
        super().__init__(**kwargs)
        self.fm_ratio = fm_ratio
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_t_start = cfg_t_start
        self.cfg_t_end = cfg_t_end
        self._needs_grad_sync = False
        self._fm_rng = random.Random(42)  # shared across ranks for deterministic FM/MF schedule

    #######################################################
    #               Training Utils                        #
    #######################################################
    def adaptive_weight(self, loss):
        return loss / (loss.detach() + self.norm_eps) ** self.norm_p

    def guided_velocity(self, net, z_t, t, v_t, model_kwargs, model_kwargs_null):
        """MeanFlow-style CFG: bake fixed-omega guidance into the target velocity.
        Returns v_t unchanged (no extra forwards) when guidance is disabled.
        Off-state matches the reference: omega == 1.0 AND kappa == 0.0 (kappa alone still guides)."""
        if self.cfg_omega == 1.0 and self.cfg_kappa == 0.0:
            return v_t
        h = th.zeros_like(t)
        with th.no_grad():
            v_u = self.u_fn(net, z_t, t, h, **model_kwargs_null)[0]
            if self.cfg_kappa == 0.0:
                v_g = v_u + self.cfg_omega * (v_t - v_u)
            else:
                v_c = self.u_fn(net, z_t, t, h, **model_kwargs)[0]
                v_g = self.cfg_omega * v_t + (1.0 - self.cfg_omega - self.cfg_kappa) * v_u + self.cfg_kappa * v_c
        active = (t >= self.cfg_t_start) & (t <= self.cfg_t_end)
        return th.where(_expand_t(active, v_t), v_g, v_t)

    def u_fn(self, model, x, t, h, return_intermediate=False, **model_kwargs):
        # MeanFlow conditions on two times: pass absolute t alongside the interval h = t - r
        model_output = model(x, h, t_abs=t, return_intermediate=return_intermediate, **model_kwargs)
        if return_intermediate:
            model_output, zt_preds = model_output
        if isinstance(model_output, tuple):
            model_output, base_output = model_output
            base_output = self.convert_model_pred(base_output, x, t)
        else:
            # None can't be used for compiled JVP, use an empty tensor instead
            base_output = th.empty(0, device=x.device)
        model_output = self.convert_model_pred(model_output, x, t)
        if return_intermediate:
            return (model_output, base_output), zt_preds
        return model_output, base_output

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################
    def training_losses(self, model, x1, model_kwargs={}, model_kwargs_null={}, z_clean=None, repa_coeff=None, base_model_coeff=1.0, percep_loss=None, cfg_dropout_prob=0.1, ema_model=None, cls_clean=None, reg_coeff=None):
        is_fm = self._fm_rng.random() < self.fm_ratio
        enable_repa = z_clean is not None and repa_coeff is not None

        if is_fm:
            return self._training_losses_fm(model, x1, model_kwargs, model_kwargs_null, z_clean, repa_coeff, base_model_coeff, cfg_dropout_prob, enable_repa, percep_loss)
        else:
            return self._training_losses_mf(model, x1, model_kwargs, model_kwargs_null, z_clean, repa_coeff, base_model_coeff, cfg_dropout_prob, enable_repa, percep_loss)

    def _training_losses_fm(self, model, x1, model_kwargs, model_kwargs_null, z_clean, repa_coeff, base_model_coeff, cfg_dropout_prob, enable_repa, percep_loss):
        """FM path: 1 fwd/bwd through DDP at h=0, supervised by the plain velocity v_t. No JVP."""
        self._needs_grad_sync = False

        t, x0, x1 = Transport.sample(self, x1)
        z_t = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        v_t = (z_t - x1) / _expand_t(t, z_t).clamp_min(self.t_eps)

        # Guided target (no-op when CFG is off); dropped samples revert to plain v_t
        v_g = self.guided_velocity(model.module, z_t, t, v_t, model_kwargs, model_kwargs_null)
        model_kwargs_dropped, drop_mask = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)
        v_g = th.where(_expand_t(drop_mask, v_t), v_t, v_g)

        # Forward through DDP (h=0 for FM); pass absolute t for two-time conditioning
        h = th.zeros_like(t)
        zt_pred = None
        if enable_repa:
            output, zt_pred = model(z_t, h, t_abs=t, return_intermediate=True, **model_kwargs_dropped)
        else:
            output = model(z_t, h, t_abs=t, **model_kwargs_dropped)

        # Handle IG dual output
        base_output = None
        if isinstance(output, tuple) and len(output) == 2:
            output, base_output = output

        output = self.convert_model_pred(output, z_t, t)
        loss = (output - v_g.detach()) ** 2
        loss = loss.sum(dim=(1, 2, 3))
        orig_loss = loss
        loss = self.adaptive_weight(loss)
        terms = {"orig_mf_loss": orig_loss, "dudt_norm": th.zeros(1, device=x1.device)}

        if percep_loss is not None:
            assert self.prediction == "x"
            # Mask based on t < percep_loss_t_thresh
            mask = t < self.percep_loss_t_thresh
            # V -> X
            terms['loss_percep'] = percep_loss(z_t - _expand_t(t, z_t) * output, x1) * mask  # [B]

        # IG base head supervision: (base_output - v_t)^2
        if base_output is not None:
            base_output = self.convert_model_pred(base_output, z_t, t)
            loss_base = (base_output - v_t) ** 2
            loss_base = loss_base.sum(dim=(1, 2, 3))
            terms['orig_loss_base'] = loss_base
            loss_base = self.adaptive_weight(loss_base) * base_model_coeff
            loss = loss + loss_base

        terms['loss'] = loss
        loss_repa = th.tensor(0.0, device=x1.device)
        if enable_repa and zt_pred is not None:
            loss_repa = repa_coeff * F.mse_loss(zt_pred, z_clean)
        terms['loss_repa'] = loss_repa
        return terms

    def _training_losses_mf(self, model, x1, model_kwargs, model_kwargs_null, z_clean, repa_coeff, base_model_coeff, cfg_dropout_prob, enable_repa, percep_loss):
        """MF path: JVP along the plain velocity field, all samples MF (no fm_mask waste)."""
        self._needs_grad_sync = True

        # Sample t, r without fm_mask — all samples are MF
        B = x1.shape[0]
        x0 = th.randn_like(x1)
        t = self.time_sampler(B).to(x1)
        r = self.time_sampler(B).to(x1)
        t, r = th.max(t, r), th.min(t, r)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        r = self.time_dist_shift * r / (1 + (self.time_dist_shift - 1) * r)

        z_t = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        v_t = (z_t - x1) / _expand_t(t, z_t).clamp_min(self.t_eps)
        zt_pred = None

        # Guided target (no-op when CFG is off); dropped samples revert to plain v_t
        v_g = self.guided_velocity(model.module, z_t, t, v_t, model_kwargs, model_kwargs_null)
        model_kwargs_dropped, drop_mask = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)
        v_g = th.where(_expand_t(drop_mask, v_t), v_t, v_g)

        def u_fn(z_t, t, r):
            result = self.u_fn(model.module, z_t, t, t - r, return_intermediate=enable_repa, **model_kwargs_dropped)
            if enable_repa:
                (model_output, base_output), zt_pred = result
                return model_output, (base_output, zt_pred)
            else:
                return result

        # JVP tangent is the (guided) velocity field (dz_t/dt = v_g), dt/dt = 1, dr/dt = 0
        with th.nn.attention.sdpa_kernel(th.nn.attention.SDPBackend.MATH):
            u_pred, du_dt, aux = th.func.jvp(u_fn, (z_t, t, r), (v_g, th.ones_like(t), th.zeros_like(r)), has_aux=True)

        if enable_repa:
            base_pred, zt_pred = aux
        else:
            base_pred = aux

        V = u_pred + _expand_t(t - r, z_t) * du_dt.detach()
        loss = (V - v_g.detach()) ** 2
        loss = loss.sum(dim=(1, 2, 3))
        orig_loss = loss
        loss = self.adaptive_weight(loss)
        terms = {
            "orig_mf_loss": orig_loss,
            "dudt_norm": du_dt.detach().norm(p=2, dim=(1, 2, 3)),
        }

        if percep_loss is not None:
            assert self.prediction == "x"
            # Mask based on t < percep_loss_t_thresh
            mask = t < self.percep_loss_t_thresh
            # U -> X
            terms['loss_percep'] = percep_loss(z_t - _expand_t(t, z_t) * u_pred, x1) * mask  # [B]

        # IG base head supervision: (base_pred - v_t)^2
        if base_pred.numel() > 0:
            loss_base = (base_pred - v_t) ** 2
            loss_base = loss_base.sum(dim=(1, 2, 3))
            terms['orig_loss_base'] = loss_base
            loss_base = self.adaptive_weight(loss_base) * base_model_coeff
            loss = loss + loss_base

        terms['loss'] = loss
        loss_repa = th.tensor(0.0, device=x1.device)
        if enable_repa and zt_pred is not None:
            loss_repa = repa_coeff * F.mse_loss(zt_pred, z_clean)
        terms['loss_repa'] = loss_repa
        return terms

    def post_backward(self, model):
        if self._needs_grad_sync:
            synchronize_gradients(model)

    def get_drift(self):
        def body_fn(x, t, h, model, **model_kwargs):
            model_output, _ = self.u_fn(model, x, t, h, **model_kwargs)
            return model_output
        return body_fn
