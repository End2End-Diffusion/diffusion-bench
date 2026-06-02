import dataclasses

from .sampler import Sampler
from .transport import Transport, TransportMF


def create_transport(config, time_dist_shift=1.0, time_dist_shift_eval=1.0):
    """Create a Transport or TransportMF from a TransportConfig."""
    shared = dict(
        prediction=config.prediction,
        time_dist_type=config.time_dist_type,
        time_dist_shift=time_dist_shift,
        time_dist_shift_eval=time_dist_shift_eval,
        t_eps=config.t_eps,
        percep_loss_t_thresh=config.percep_loss_t_thresh,
    )
    if config.meanflow is not None:
        return TransportMF(**shared, **dataclasses.asdict(config.meanflow))
    return Transport(**shared)


def create_sampler(transport, guidance_config):
    return Sampler(transport, guidance_config=guidance_config)


__all__ = ["create_transport", "create_sampler", "Transport", "TransportMF", "Sampler"]
