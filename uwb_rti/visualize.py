import numpy as np
import matplotlib.pyplot as plt
from .config import GRID_SIZE, AREA_SIZE, TX_POSITIONS, RX_POSITIONS


def plot_weight_matrix(W, link_indices, save_path="figures/weight_matrix.png"):
    """Plot weight vectors reshaped to 30x30 for selected links."""
    n = len(link_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, link_indices):
        tx_i = idx // len(RX_POSITIONS)
        rx_i = idx % len(RX_POSITIONS)
        img = W[idx].reshape(GRID_SIZE, GRID_SIZE)
        im = ax.imshow(img, extent=[0, AREA_SIZE, 0, AREA_SIZE], origin="lower", cmap="hot")
        ax.set_title(f"Link {idx} (TX{tx_i}->RX{rx_i})")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        tx = TX_POSITIONS[tx_i]
        rx = RX_POSITIONS[rx_i]
        ax.plot(*tx, "bv", markersize=10, label="TX")
        ax.plot(*rx, "r^", markersize=10, label="RX")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved weight matrix plot to {save_path}")


def plot_rss_vs_distance(distances, rss_values, save_path="figures/rss_vs_distance.png"):
    """Plot RSS vs distance for model validation."""
    plt.figure(figsize=(8, 5))
    plt.plot(distances, rss_values, "b-o", linewidth=2, markersize=4)
    plt.xlabel("Distance (m)")
    plt.ylabel("RSS (dB)")
    plt.title("RSS vs Distance (No Shadowing)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved RSS vs distance plot to {save_path}")


def plot_rss_vs_position(positions, rss_values, save_path="figures/rss_vs_position.png"):
    """Plot RSS vs person x-position for human crossing validation."""
    plt.figure(figsize=(8, 5))
    plt.plot(positions, rss_values, "r-o", linewidth=2, markersize=4)
    plt.xlabel("Person x-position (m)")
    plt.ylabel("RSS (dB)")
    plt.title("RSS During Human Crossing (TX-RX LOS)")
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="TX-RX line")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved RSS vs position plot to {save_path}")


def plot_training_curves(history, title, save_path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_reconstruction_grid(
    ground_truths, mlp_outputs, cfp_outputs, num_samples=5,
    save_path="figures/reconstruction_grid.png",
):
    """Plot ground truth vs MLP vs MLP+CFP for several samples."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    col_titles = ["Ground Truth", "MLP Output", "MLP+CFP Output"]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=14)

    for i in range(num_samples):
        gt = ground_truths[i].reshape(GRID_SIZE, GRID_SIZE)
        mlp = mlp_outputs[i].reshape(GRID_SIZE, GRID_SIZE)
        cfp = cfp_outputs[i].reshape(GRID_SIZE, GRID_SIZE)

        vmin = min(gt.min(), mlp.min(), cfp.min())
        vmax = max(gt.max(), mlp.max(), cfp.max())

        for col, img in enumerate([gt, mlp, cfp]):
            im = axes[i, col].imshow(
                img, extent=[0, AREA_SIZE, 0, AREA_SIZE],
                origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
            )
            axes[i, col].set_xlabel("x (m)")
            axes[i, col].set_ylabel("y (m)")

        fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction grid to {save_path}")


def plot_error_maps(
    ground_truths, mlp_outputs, cfp_outputs, num_samples=5,
    save_path="figures/error_maps.png",
):
    """Plot |predicted - ground truth| error maps."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    axes[0, 0].set_title("MLP Error", fontsize=14)
    axes[0, 1].set_title("MLP+CFP Error", fontsize=14)

    for i in range(num_samples):
        gt = ground_truths[i].reshape(GRID_SIZE, GRID_SIZE)
        mlp_err = np.abs(mlp_outputs[i].reshape(GRID_SIZE, GRID_SIZE) - gt)
        cfp_err = np.abs(cfp_outputs[i].reshape(GRID_SIZE, GRID_SIZE) - gt)

        vmax = max(mlp_err.max(), cfp_err.max())

        for col, err in enumerate([mlp_err, cfp_err]):
            im = axes[i, col].imshow(
                err, extent=[0, AREA_SIZE, 0, AREA_SIZE],
                origin="lower", cmap="hot", vmin=0, vmax=vmax,
            )
            axes[i, col].set_xlabel("x (m)")
            axes[i, col].set_ylabel("y (m)")

        fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error maps to {save_path}")


def plot_noise_comparison_table(results, save_path="figures/noise_comparison.png"):
    """Plot quantitative comparison table across noise levels.

    Args:
        results: dict with keys as noise levels (sigma_eps), values as dicts
                 with keys 'mlp_rmse', 'mlp_ssim', 'cfp_rmse', 'cfp_ssim'.
    """
    noise_levels = sorted(results.keys())
    col_labels = ["sigma_eps", "MLP RMSE", "MLP SSIM", "MLP+CFP RMSE", "MLP+CFP SSIM"]
    table_data = []
    for sigma in noise_levels:
        r = results[sigma]
        table_data.append([
            f"{sigma:.1f}",
            f"{r['mlp_rmse']:.4f}",
            f"{r['mlp_ssim']:.4f}",
            f"{r['cfp_rmse']:.4f}",
            f"{r['cfp_ssim']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(noise_levels)))
    ax.axis("off")
    table = ax.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax.set_title("RMSE and SSIM Across Noise Levels", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved noise comparison table to {save_path}")
