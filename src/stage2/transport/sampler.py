import torch as th


class Sampler:
    def __init__(self, transport, guidance_config):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.guidance_config = guidance_config

    def sample_ode(self, *, num_steps=50):
        t_grid = th.linspace(1.0, 0.0, num_steps + 1)
        shift = self.transport.time_dist_shift_eval
        t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

        def sample_fn(x, model, **model_kwargs):
            device = x.device
            t_steps = t_grid.to(device)
            B = x.shape[0]

            cls = model_kwargs.pop('cls_t', None)  # reg models evolve a cls token alongside x

            for i in range(num_steps):
                h = t_steps[i] - t_steps[i + 1]
                h_batch = th.full((B,), h.item(), device=device)
                t_batch = th.full((B,), t_steps[i].item(), device=device)
                if cls is not None:
                    d_x, d_cls = self.drift(x, t_batch, h_batch, model, cls_t=cls, **model_kwargs)
                    x = x - h * d_x
                    cls = cls - h * d_cls
                else:
                    d_cur = self.drift(x, t_batch, h_batch, model, **model_kwargs)
                    x = x - h * d_cur

            return x.unsqueeze(0)

        return sample_fn
