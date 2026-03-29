import numpy as np
from .config import (
    TX_POSITIONS, RX_POSITIONS, NUM_LINKS, NUM_PIXELS,
    PIXEL_CENTERS, LAMBDA, BETA_MIN, SCALING_C,
)


def compute_weight_matrix():
    """Compute weight matrix W (NUM_LINKS x NUM_PIXELS) using inverse area elliptical model."""
    W = np.zeros((NUM_LINKS, NUM_PIXELS))

    link_idx = 0
    for tx in TX_POSITIONS:
        for rx in RX_POSITIONS:
            d_nm = np.linalg.norm(tx - rx)
            beta_max = np.sqrt(LAMBDA * d_nm / 4)

            d1 = np.linalg.norm(PIXEL_CENTERS - tx, axis=1)
            d2 = np.linalg.norm(PIXEL_CENTERS - rx, axis=1)
            a = (d1 + d2) / 2
            c_half = d_nm / 2

            arg = a**2 - c_half**2
            beta = np.sqrt(np.maximum(arg, 0.0))

            weights = np.zeros(NUM_PIXELS)
            mask_clamped = beta <= BETA_MIN
            mask_normal = (~mask_clamped) & (beta < beta_max)

            weights[mask_clamped] = 1.0 / (np.pi * a[mask_clamped] * BETA_MIN)
            weights[mask_normal] = 1.0 / (np.pi * a[mask_normal] * beta[mask_normal])

            W[link_idx] = weights
            link_idx += 1

    return W


def compute_distance_vector():
    """Compute log-distance vector d (NUM_LINKS,) for all TX-RX pairs."""
    d = np.zeros(NUM_LINKS)
    link_idx = 0
    for tx in TX_POSITIONS:
        for rx in RX_POSITIONS:
            D = np.linalg.norm(tx - rx)
            d[link_idx] = 20.0 * np.log10(D)
            link_idx += 1
    return d


def compute_rss(theta, b, alpha, sigma_eps, W, d, rng=None):
    """Compute RSS measurement vector y (NUM_LINKS,).

    Args:
        theta: SLF image vector (NUM_PIXELS,)
        b: bias vector (NUM_LINKS,)
        alpha: path loss exponent (scalar)
        sigma_eps: noise standard deviation (scalar)
        W: weight matrix (NUM_LINKS, NUM_PIXELS)
        d: log-distance vector (NUM_LINKS,)
        rng: optional numpy random Generator for reproducibility
    """
    if rng is not None:
        epsilon = rng.normal(0, sigma_eps, size=NUM_LINKS)
    else:
        epsilon = np.random.normal(0, sigma_eps, size=NUM_LINKS)
    y = b - SCALING_C * (W @ theta) - alpha * d + epsilon
    return y
