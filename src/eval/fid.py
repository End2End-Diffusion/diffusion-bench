import numpy as np
import scipy.linalg
import torch
from torch_fidelity import calculate_metrics
from torch_fidelity.feature_extractor_inceptionv3 import (
    FeatureExtractorInceptionV3,  # original torch-fidelity :contentReference[oaicite:2]{index=2}
)

from .utils import ImgArrDataset


def _fid_from_moments(mu1, sigma1, mu2, sigma2) -> float:
    # FID formula :contentReference[oaicite:3]{index=3}

    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):  # numerical noise
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


@torch.no_grad()
def _compute_inception_moments_from_arr(arr: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses torch-fidelity's InceptionV3 feature extractor to get 2048-d pool features.
    Assumes arr is [N,H,W,C] or [N,C,H,W], uint8 (0..255) or float (0..1 or 0..255).
    """
    # Convert to torch in NCHW uint8 as the safest default.
    x = arr
    if x.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {x.shape}")

    if x.shape[-1] == 3:  # NHWC -> NCHW
        x = np.transpose(x, (0, 3, 1, 2))

    if x.dtype != np.uint8:
        # If float in [0,1], scale up; otherwise assume already 0..255-ish
        x_f = x.astype(np.float32)
        if x_f.max() <= 1.5:
            x_f = x_f * 255.0
        x = np.clip(x_f, 0, 255).astype(np.uint8)

    xt = torch.from_numpy(x).to(device=device, dtype=torch.uint8)

    fe = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=['2048', 'logits_unbiased']).to(device).eval()  # preregistered extractor name :contentReference[oaicite:4]{index=4}

    feats = []
    logits = []
    for i in range(0, xt.shape[0], batch_size):
        batch = xt[i : i + batch_size]
        f, f_logits = fe(batch)
        feats.append(f.detach().cpu())
        logits.append(f_logits.detach().cpu())

    feats = torch.cat(feats, dim=0).double().numpy()  # (N, 2048) float64
    logits = torch.cat(logits, dim=0)  # (N, 1000) keep as tensor for softmax
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma, logits


def _inception_score(logits, num_splits=10):
    """Compute Inception Score (mean) from raw logits."""
    probs = torch.nn.functional.softmax(logits.float(), dim=1).numpy()
    n = len(probs)
    probs = probs[np.random.RandomState(0).permutation(n)]
    split_size = n // num_splits
    scores = []
    for i in range(num_splits):
        part = probs[i * split_size : (i + 1) * split_size]
        p_y = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(scores))


def calculate_gfid(
    arr1: np.ndarray,
    ref_arr: dict,
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[float, float]:
    if 'mu' in ref_arr and 'sigma' in ref_arr:
        mu_ref, sigma_ref = ref_arr['mu'], ref_arr['sigma']
    elif 'ref_mu' in ref_arr and 'ref_sigma' in ref_arr:
        mu_ref, sigma_ref = ref_arr['ref_mu'], ref_arr['ref_sigma']
    else:
        raise KeyError(f"Reference has no mu/sigma or ref_mu/ref_sigma keys, found: {list(ref_arr.keys())}")
    mu_gen, sigma_gen, logits_gen = _compute_inception_moments_from_arr(arr1, batch_size=batch_size, device=device)
    return _fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref), _inception_score(logits_gen)

def calculate_rfid(
    arr1,
    arr2=None,
    bs=64,
    device="cuda",
    fid_statistics_file=None,
):
    arr1_ds = ImgArrDataset(arr1)

    if fid_statistics_file is not None:
        metrics_kwargs = dict(
            input1=arr1_ds,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )
    else:
        if arr2 is None:
            raise ValueError("Either arr2 or fid_statistics_file must be provided.")
        arr2_ds = ImgArrDataset(arr2)
        metrics_kwargs = dict(
            input1=arr1_ds,
            input2=arr2_ds,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )

    metrics = calculate_metrics(**metrics_kwargs)
    return metrics["frechet_inception_distance"]
