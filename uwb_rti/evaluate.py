import os
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from .config import (
    DEVICE, GRID_SIZE, NUM_LINKS, NUM_PIXELS,
    SCALING_C, RANDOM_SEED, BATCH_SIZE,
)


def compute_rmse(pred, true):
    """Compute per-sample RMSE, then average."""
    per_sample = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    return np.mean(per_sample)


def compute_ssim(pred, true):
    """Compute per-sample SSIM on 30x30 images, then average."""
    n = pred.shape[0]
    ssim_values = np.zeros(n)
    for i in range(n):
        p = pred[i].reshape(GRID_SIZE, GRID_SIZE)
        t = true[i].reshape(GRID_SIZE, GRID_SIZE)
        data_range = max(t.max() - t.min(), 1e-8)
        ssim_values[i] = ssim(t, p, data_range=data_range)
    return np.mean(ssim_values)


def run_inference(mlp_model, cfp_model, rss):
    """Run MLP (and optionally CFP) inference on RSS data in batches."""
    mlp_model.eval()
    n = rss.shape[0]
    mlp_out = np.zeros((n, NUM_PIXELS))

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = torch.FloatTensor(rss[start:end]).to(DEVICE)
            mlp_out[start:end] = mlp_model(batch).cpu().numpy()

    cfp_out = None
    if cfp_model is not None:
        cfp_model.eval()
        cfp_out = np.zeros((n, NUM_PIXELS))
        mlp_images = mlp_out.reshape(-1, 1, GRID_SIZE, GRID_SIZE)
        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                batch = torch.FloatTensor(mlp_images[start:end]).to(DEVICE)
                cfp_out[start:end] = cfp_model(batch).cpu().numpy().reshape(-1, NUM_PIXELS)

    return mlp_out, cfp_out


def evaluate_on_test_set(mlp_model, cfp_model, data_dir="data"):
    """Evaluate models on the held-out test set."""
    data = np.load(os.path.join(data_dir, "test_data.npz"))
    rss = data["rss"]
    theta_ideal = data["theta_ideal"]

    mlp_out, cfp_out = run_inference(mlp_model, cfp_model, rss)

    results = {
        "mlp_rmse": compute_rmse(mlp_out, theta_ideal),
        "mlp_ssim": compute_ssim(mlp_out, theta_ideal),
    }
    if cfp_out is not None:
        results["cfp_rmse"] = compute_rmse(cfp_out, theta_ideal)
        results["cfp_ssim"] = compute_ssim(cfp_out, theta_ideal)

    print("\n=== Test Set Results ===")
    print(f"MLP      - RMSE: {results['mlp_rmse']:.4f}, SSIM: {results['mlp_ssim']:.4f}")
    if cfp_out is not None:
        print(f"MLP+CFP  - RMSE: {results['cfp_rmse']:.4f}, SSIM: {results['cfp_ssim']:.4f}")

    return results, mlp_out, cfp_out, theta_ideal


def evaluate_across_noise_levels(
    mlp_model, cfp_model, data_dir="data",
    noise_levels=(0.5, 1.0, 2.0, 3.0),
):
    """Evaluate at fixed noise levels using the same test SLF images.

    Regenerates RSS measurements at each noise level from the saved
    forward model and test SLF images.
    """
    test_data = np.load(os.path.join(data_dir, "test_data.npz"))
    theta_ideal = test_data["theta_ideal"]
    theta_noisy = test_data["theta_noisy"]

    fm_data = np.load(os.path.join(data_dir, "forward_model.npz"))
    W = fm_data["W"]
    d = fm_data["d"]

    norm_data = np.load(os.path.join(data_dir, "norm_stats.npz"))
    rss_mean = norm_data["rss_mean"]
    rss_std = norm_data["rss_std"]

    rng = np.random.default_rng(RANDOM_SEED + 1000)
    results = {}

    for sigma_eps in noise_levels:
        rss_all = np.zeros((len(theta_noisy), NUM_LINKS))
        for i in range(len(theta_noisy)):
            b = rng.uniform(90.0, 100.0, size=NUM_LINKS)
            alpha = rng.uniform(0.9, 1.0)
            epsilon = rng.normal(0, sigma_eps, size=NUM_LINKS)
            rss_all[i] = b - SCALING_C * (W @ theta_noisy[i]) - alpha * d + epsilon

        rss_norm = (rss_all - rss_mean) / rss_std

        mlp_out, cfp_out = run_inference(mlp_model, cfp_model, rss_norm)

        r = {
            "mlp_rmse": compute_rmse(mlp_out, theta_ideal),
            "mlp_ssim": compute_ssim(mlp_out, theta_ideal),
        }
        if cfp_out is not None:
            r["cfp_rmse"] = compute_rmse(cfp_out, theta_ideal)
            r["cfp_ssim"] = compute_ssim(cfp_out, theta_ideal)

        results[sigma_eps] = r
        print(f"sigma_eps={sigma_eps:.1f} - MLP RMSE: {r['mlp_rmse']:.4f}, "
              f"MLP SSIM: {r['mlp_ssim']:.4f}", end="")
        if cfp_out is not None:
            print(f", CFP RMSE: {r['cfp_rmse']:.4f}, CFP SSIM: {r['cfp_ssim']:.4f}")
        else:
            print()

    return results
