import os
import numpy as np
from .config import (
    GRID_SIZE, NUM_PIXELS, NUM_LINKS, PIXEL_SIZE, AREA_SIZE,
    PIXEL_CENTERS, KAPPA, BIAS_RANGE, ALPHA_RANGE,
    NOISE_SIGMA_RANGE, SLF_NOISE_SIGMA_RANGE, SCALING_C, RANDOM_SEED,
    MLP_DATASET_SIZE, CFP_DATASET_SIZE, TEST_DATASET_SIZE,
    MLP_TRAIN_SIZE, MLP_VAL_SIZE,
)
from .forward_model import compute_weight_matrix, compute_distance_vector


def _place_rectangle(theta, cx, cy, width, height, attenuation):
    """Place a rectangular object on the SLF grid."""
    mask = (
        (PIXEL_CENTERS[:, 0] >= cx - width / 2)
        & (PIXEL_CENTERS[:, 0] <= cx + width / 2)
        & (PIXEL_CENTERS[:, 1] >= cy - height / 2)
        & (PIXEL_CENTERS[:, 1] <= cy + height / 2)
    )
    theta[mask] = attenuation


def _place_circle(theta, cx, cy, radius, attenuation):
    """Place a circular object on the SLF grid."""
    dist_sq = (PIXEL_CENTERS[:, 0] - cx) ** 2 + (PIXEL_CENTERS[:, 1] - cy) ** 2
    theta[dist_sq <= radius**2] = attenuation


def _random_center(rng, obj_width, obj_height, margin=0.2):
    """Generate a random center position ensuring object stays within area."""
    cx = rng.uniform(margin + obj_width / 2, AREA_SIZE - margin - obj_width / 2)
    cy = rng.uniform(margin + obj_height / 2, AREA_SIZE - margin - obj_height / 2)
    return cx, cy


def _place_single_object(theta, obj_type, rng):
    """Place one object of the given type."""
    if obj_type == 0:  # Standing person
        w, h = 0.4, 0.4
        cx, cy = _random_center(rng, w, h)
        _place_rectangle(theta, cx, cy, w, h, rng.uniform(0.5, 1.0))
    elif obj_type == 1:  # Walking person
        w, h = 0.3, 0.5
        cx, cy = _random_center(rng, w, h)
        _place_rectangle(theta, cx, cy, w, h, rng.uniform(0.5, 1.0))
    elif obj_type == 2:  # Table/Desk
        w, h = 0.8, 0.6
        cx, cy = _random_center(rng, w, h)
        _place_rectangle(theta, cx, cy, w, h, rng.uniform(0.3, 0.6))
    elif obj_type == 3:  # Chair
        w, h = 0.4, 0.4
        cx, cy = _random_center(rng, w, h)
        _place_rectangle(theta, cx, cy, w, h, rng.uniform(0.3, 0.5))
    elif obj_type == 4:  # Cabinet/Shelf
        w, h = 0.5, 0.3
        cx, cy = _random_center(rng, w, h)
        _place_rectangle(theta, cx, cy, w, h, rng.uniform(0.5, 0.8))
    elif obj_type == 5:  # Wall segment
        w, h = 0.1, 1.0
        cx, cy = _random_center(rng, w, h)
        _place_rectangle(theta, cx, cy, w, h, rng.uniform(0.6, 1.0))


def generate_slf_image(rng):
    """Generate a random ideal SLF image theta* (NUM_PIXELS,)."""
    theta = np.zeros(NUM_PIXELS)
    obj_type = rng.integers(0, 10)

    if obj_type == 7:  # Empty room
        return theta

    if obj_type == 6:  # Multiple objects (2-3)
        num_objects = rng.integers(2, 4)
        for _ in range(num_objects):
            sub_type = rng.integers(0, 6)
            _place_single_object(theta, sub_type, rng)
        return theta

    if obj_type == 8:  # L-shaped or T-shaped
        att = rng.uniform(0.5, 1.0)
        shape = rng.integers(0, 2)  # 0 = L, 1 = T
        if shape == 0:  # L-shaped
            cx, cy = _random_center(rng, 0.8, 0.9)
            _place_rectangle(theta, cx, cy, 0.8, 0.3, att)
            _place_rectangle(theta, cx - 0.25, cy + 0.3, 0.3, 0.6, att)
        else:  # T-shaped
            cx, cy = _random_center(rng, 0.8, 0.9)
            _place_rectangle(theta, cx, cy, 0.8, 0.3, att)
            _place_rectangle(theta, cx, cy - 0.3, 0.3, 0.6, att)
        return theta

    if obj_type == 9:  # Circular pillar
        radius = rng.uniform(0.2, 0.3)
        cx, cy = _random_center(rng, 2 * radius, 2 * radius)
        _place_circle(theta, cx, cy, radius, rng.uniform(0.5, 1.0))
        return theta

    _place_single_object(theta, obj_type, rng)
    return theta


def compute_cholesky_factor():
    """Pre-compute Cholesky factor of base spatial correlation matrix."""
    diff = PIXEL_CENTERS[:, np.newaxis, :] - PIXEL_CENTERS[np.newaxis, :, :]
    D_kl = np.sqrt(np.sum(diff**2, axis=2))
    C_base = np.exp(-D_kl / KAPPA)
    # Add small jitter for numerical stability
    C_base += 1e-10 * np.eye(NUM_PIXELS)
    L = np.linalg.cholesky(C_base)
    return L


def generate_dataset(num_samples, W, d, L, seed):
    """Generate dataset of (RSS, noisy SLF, ideal SLF) triples."""
    rng = np.random.default_rng(seed)

    rss_all = np.zeros((num_samples, NUM_LINKS))
    theta_noisy_all = np.zeros((num_samples, NUM_PIXELS))
    theta_ideal_all = np.zeros((num_samples, NUM_PIXELS))

    for i in range(num_samples):
        theta_star = generate_slf_image(rng)

        sigma_theta = rng.uniform(*SLF_NOISE_SIGMA_RANGE)
        z = rng.standard_normal(NUM_PIXELS)
        theta_tilde = sigma_theta * (L @ z)
        theta = theta_star + theta_tilde

        b = rng.uniform(*BIAS_RANGE, size=NUM_LINKS)
        alpha = rng.uniform(*ALPHA_RANGE)
        sigma_eps = rng.uniform(*NOISE_SIGMA_RANGE)
        epsilon = rng.normal(0, sigma_eps, size=NUM_LINKS)

        y = b - SCALING_C * (W @ theta) - alpha * d + epsilon

        rss_all[i] = y
        theta_noisy_all[i] = theta
        theta_ideal_all[i] = theta_star

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    return rss_all, theta_noisy_all, theta_ideal_all


def generate_and_save_all_datasets(data_dir="data"):
    """Generate all datasets and save to disk."""
    os.makedirs(data_dir, exist_ok=True)

    print("Computing weight matrix, distance vector, and Cholesky factor...")
    W = compute_weight_matrix()
    d = compute_distance_vector()
    L = compute_cholesky_factor()

    print(f"Generating MLP dataset ({MLP_DATASET_SIZE} samples)...")
    rss_mlp, theta_noisy_mlp, theta_ideal_mlp = generate_dataset(
        MLP_DATASET_SIZE, W, d, L, seed=RANDOM_SEED
    )

    print(f"Generating CFP dataset ({CFP_DATASET_SIZE} samples)...")
    rss_cfp, theta_noisy_cfp, theta_ideal_cfp = generate_dataset(
        CFP_DATASET_SIZE, W, d, L, seed=RANDOM_SEED + 1
    )

    print(f"Generating test dataset ({TEST_DATASET_SIZE} samples)...")
    rss_test, theta_noisy_test, theta_ideal_test = generate_dataset(
        TEST_DATASET_SIZE, W, d, L, seed=RANDOM_SEED + 2
    )

    # Compute normalization stats from MLP training split
    rss_train = rss_mlp[:MLP_TRAIN_SIZE]
    rss_mean = rss_train.mean(axis=0)
    rss_std = rss_train.std(axis=0)
    rss_std[rss_std < 1e-8] = 1.0  # Prevent division by zero

    # Normalize all RSS data
    rss_mlp = (rss_mlp - rss_mean) / rss_std
    rss_cfp = (rss_cfp - rss_mean) / rss_std
    rss_test = (rss_test - rss_mean) / rss_std

    # Save
    np.savez(
        os.path.join(data_dir, "mlp_data.npz"),
        rss=rss_mlp, theta_noisy=theta_noisy_mlp, theta_ideal=theta_ideal_mlp,
    )
    np.savez(
        os.path.join(data_dir, "cfp_data.npz"),
        rss=rss_cfp, theta_noisy=theta_noisy_cfp, theta_ideal=theta_ideal_cfp,
    )
    np.savez(
        os.path.join(data_dir, "test_data.npz"),
        rss=rss_test, theta_noisy=theta_noisy_test, theta_ideal=theta_ideal_test,
    )
    np.savez(
        os.path.join(data_dir, "norm_stats.npz"),
        rss_mean=rss_mean, rss_std=rss_std,
    )
    # Save weight matrix and distance vector for later use
    np.savez(
        os.path.join(data_dir, "forward_model.npz"),
        W=W, d=d,
    )

    print(f"All datasets saved to {data_dir}/")


if __name__ == "__main__":
    generate_and_save_all_datasets()
