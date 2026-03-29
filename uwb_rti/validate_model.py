import numpy as np
from .config import (
    AREA_SIZE, GRID_SIZE, NUM_PIXELS, PIXEL_CENTERS, PIXEL_SIZE,
    LAMBDA, BETA_MIN, SCALING_C,
)
from .visualize import plot_rss_vs_distance, plot_rss_vs_position


def _compute_weight_vector_single(tx, rx):
    """Compute weight vector for a single TX-RX pair."""
    d_nm = np.linalg.norm(tx - rx)
    beta_max = np.sqrt(LAMBDA * d_nm / 4)

    d1 = np.linalg.norm(PIXEL_CENTERS - tx, axis=1)
    d2 = np.linalg.norm(PIXEL_CENTERS - rx, axis=1)
    a = (d1 + d2) / 2
    c_half = d_nm / 2

    arg = a**2 - c_half**2
    beta = np.sqrt(np.maximum(arg, 0.0))

    w = np.zeros(NUM_PIXELS)
    mask_clamped = beta <= BETA_MIN
    mask_normal = (~mask_clamped) & (beta < beta_max)
    w[mask_clamped] = 1.0 / (np.pi * a[mask_clamped] * BETA_MIN)
    w[mask_normal] = 1.0 / (np.pi * a[mask_normal] * beta[mask_normal])
    return w


def validate_rss_vs_distance(save_path="figures/rss_vs_distance.png"):
    """Validation 1: RSS vs distance with no shadowing."""
    b = 95.0
    alpha = 0.95

    distances = np.linspace(0.5, 5.0, 50)
    rss_values = b - alpha * 20.0 * np.log10(distances)

    plot_rss_vs_distance(distances, rss_values, save_path)
    print("Validation 1 complete: RSS decreases logarithmically with distance.")


def validate_human_crossing(save_path="figures/rss_vs_position.png"):
    """Validation 2: RSS change during human crossing."""
    tx = np.array([0.0, 0.0])
    rx = np.array([0.0, 3.0])
    D = np.linalg.norm(tx - rx)
    d_log = 20.0 * np.log10(D)

    b = 95.0
    alpha = 0.95

    w = _compute_weight_vector_single(tx, rx)

    person_width = 0.4
    person_height = 0.4
    person_attenuation = 0.7
    person_y = 1.5

    x_positions = np.linspace(-1.0, 4.0, 100)
    rss_values = np.zeros_like(x_positions)

    for i, px in enumerate(x_positions):
        theta = np.zeros(NUM_PIXELS)
        mask = (
            (PIXEL_CENTERS[:, 0] >= px - person_width / 2)
            & (PIXEL_CENTERS[:, 0] <= px + person_width / 2)
            & (PIXEL_CENTERS[:, 1] >= person_y - person_height / 2)
            & (PIXEL_CENTERS[:, 1] <= person_y + person_height / 2)
        )
        theta[mask] = person_attenuation

        s = SCALING_C * w @ theta
        rss_values[i] = b - s - alpha * d_log

    plot_rss_vs_position(x_positions, rss_values, save_path)
    print("Validation 2 complete: RSS dips when person crosses LOS path.")


def run_all_validations():
    """Run all forward model validations."""
    import os
    os.makedirs("figures", exist_ok=True)
    validate_rss_vs_distance()
    validate_human_crossing()


if __name__ == "__main__":
    run_all_validations()
