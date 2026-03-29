# UWB RTI Deep Learning System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a complete UWB RTI system with forward model, data generation, MLP and CFP deep learning models, training, evaluation, and visualization.

**Architecture:** Sequential pipeline — forward model produces weight matrix and RSS measurements, data generator creates 61k training/test samples, MLP reconstructs SLF images from RSS vectors, CFP post-processes MLP outputs. All code in a `uwb_rti` Python package with relative imports, run via `python -m uwb_rti.main`.

**Tech Stack:** Python 3, PyTorch (CUDA), NumPy, matplotlib, scikit-image

---

## File Structure

```
uwb-rti-with-DL/
├── requirements.txt
├── uwb_rti/
│   ├── __init__.py
│   ├── config.py              # All constants and parameters
│   ├── forward_model.py       # Weight matrix, RSS generation
│   ├── data_generator.py      # Training data generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp_model.py       # MLP architecture
│   │   └── cfp_model.py       # CFP CNN architecture
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation and metrics
│   ├── validate_model.py      # Forward model validation
│   ├── visualize.py           # All visualization functions
│   └── main.py                # Main execution pipeline
├── data/                      # Generated datasets (.npz)
├── checkpoints/               # Model checkpoints (.pt)
└── figures/                   # Saved plots (.png)
```

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `uwb_rti/__init__.py`
- Create: `uwb_rti/models/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
torch
numpy
matplotlib
scikit-image
```

- [ ] **Step 2: Create package init files**

`uwb_rti/__init__.py` — empty file.

`uwb_rti/models/__init__.py` — empty file.

- [ ] **Step 3: Create output directories**

```bash
mkdir -p data checkpoints figures
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt uwb_rti/__init__.py uwb_rti/models/__init__.py
git commit -m "feat: project setup with dependencies and package structure"
```

---

### Task 2: Configuration Module

**Files:**
- Create: `uwb_rti/config.py`

- [ ] **Step 1: Write config.py**

```python
import numpy as np
import torch

# Random seed for reproducibility
RANDOM_SEED = 42

# Physical constants
C_LIGHT = 3e8
F_CENTER = 7.9872e9
BANDWIDTH = 499.2e6
LAMBDA = C_LIGHT / F_CENTER

# Geometry
AREA_SIZE = 3.0
GRID_SIZE = 30
NUM_PIXELS = GRID_SIZE * GRID_SIZE  # 900
PIXEL_SIZE = AREA_SIZE / GRID_SIZE

# Node positions
NUM_TX = 4
NUM_RX = 4
NUM_LINKS = NUM_TX * NUM_RX  # 16

TX_POSITIONS = np.array([
    [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]
])

RX_POSITIONS = np.array([
    [0.0, 3.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0]
])

# Pixel center positions
_cols = np.arange(GRID_SIZE)
_rows = np.arange(GRID_SIZE)
_pixel_x = (_cols + 0.5) * PIXEL_SIZE
_pixel_y = (_rows + 0.5) * PIXEL_SIZE
_PX, _PY = np.meshgrid(_pixel_x, _pixel_y)
PIXEL_CENTERS = np.stack([_PX.ravel(), _PY.ravel()], axis=1)  # (900, 2)

# Weight model parameters
BETA_MIN = LAMBDA / 20
SCALING_C = 2

# Parameter ranges
BIAS_RANGE = (90.0, 100.0)
ALPHA_RANGE = (0.9, 1.0)
NOISE_SIGMA_RANGE = (0.3, 3.0)
SLF_NOISE_SIGMA_RANGE = (0.01, 0.05)
KAPPA = 0.21

# Dataset sizes
MLP_DATASET_SIZE = 40000
CFP_DATASET_SIZE = 20000
TEST_DATASET_SIZE = 1000
MLP_TRAIN_SIZE = 36000
MLP_VAL_SIZE = 4000
CFP_TRAIN_SIZE = 18000
CFP_VAL_SIZE = 2000

# Training hyperparameters
BATCH_SIZE = 256
MLP_LR = 1e-3
MLP_EPOCHS = 200
MLP_PATIENCE = 20
CFP_LR = 0.001
CFP_MOMENTUM = 0.9
CFP_EPOCHS = 200

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 2: Verify config loads**

```bash
python -c "from uwb_rti.config import *; print(f'LAMBDA={LAMBDA:.5f}m, BETA_MIN={BETA_MIN:.6f}m, NUM_PIXELS={NUM_PIXELS}, DEVICE={DEVICE}')"
```

Expected output: `LAMBDA=0.03755m, BETA_MIN=0.001878m, NUM_PIXELS=900, DEVICE=cuda`

- [ ] **Step 3: Commit**

```bash
git add uwb_rti/config.py
git commit -m "feat: add configuration module with all constants and parameters"
```

---

### Task 3: Forward Model

**Files:**
- Create: `uwb_rti/forward_model.py`

- [ ] **Step 1: Write forward_model.py**

```python
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
```

- [ ] **Step 2: Verify forward model**

```bash
python -c "
from uwb_rti.forward_model import compute_weight_matrix, compute_distance_vector
W = compute_weight_matrix()
d = compute_distance_vector()
print(f'W shape: {W.shape}, W range: [{W.min():.4f}, {W.max():.4f}]')
print(f'd shape: {d.shape}, d range: [{d.min():.2f}, {d.max():.2f}]')
print(f'Non-zero weights per link: {(W > 0).sum(axis=1)}')
"
```

Expected: W shape (16, 900), all values non-negative, d shape (16,), non-zero weight counts vary by link distance.

- [ ] **Step 3: Commit**

```bash
git add uwb_rti/forward_model.py
git commit -m "feat: add forward model with weight matrix and RSS computation"
```

---

### Task 4: Visualization Module

**Files:**
- Create: `uwb_rti/visualize.py`

- [ ] **Step 1: Write visualize.py**

```python
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
```

- [ ] **Step 2: Verify visualization imports and generate weight matrix plot**

```bash
python -c "
from uwb_rti.forward_model import compute_weight_matrix
from uwb_rti.visualize import plot_weight_matrix
import os; os.makedirs('figures', exist_ok=True)
W = compute_weight_matrix()
plot_weight_matrix(W, [0, 5, 10, 15])
"
```

Expected: `figures/weight_matrix.png` created, showing elliptical weight patterns for 4 links.

- [ ] **Step 3: Commit**

```bash
git add uwb_rti/visualize.py
git commit -m "feat: add visualization module with all plot functions"
```

---

### Task 5: Forward Model Validation

**Files:**
- Create: `uwb_rti/validate_model.py`

- [ ] **Step 1: Write validate_model.py**

```python
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
```

- [ ] **Step 2: Run validations and inspect plots**

```bash
python -m uwb_rti.validate_model
```

Expected: Two plots saved to `figures/`. Validation 1 shows monotonically decreasing RSS. Validation 2 shows RSS dip near x=0 (LOS crossing point).

- [ ] **Step 3: Commit**

```bash
git add uwb_rti/validate_model.py
git commit -m "feat: add forward model validation with RSS vs distance and human crossing"
```

---

### Task 6: Data Generator

**Files:**
- Create: `uwb_rti/data_generator.py`

- [ ] **Step 1: Write data_generator.py**

```python
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
```

- [ ] **Step 2: Run data generation**

```bash
python -m uwb_rti.data_generator
```

Expected: Generates 61k samples total. Files saved to `data/`: `mlp_data.npz`, `cfp_data.npz`, `test_data.npz`, `norm_stats.npz`, `forward_model.npz`.

- [ ] **Step 3: Spot-check generated data**

```bash
python -c "
import numpy as np
d = np.load('data/mlp_data.npz')
print(f'RSS: shape={d[\"rss\"].shape}, mean={d[\"rss\"].mean():.3f}, std={d[\"rss\"].std():.3f}')
print(f'Ideal SLF: shape={d[\"theta_ideal\"].shape}, range=[{d[\"theta_ideal\"].min():.3f}, {d[\"theta_ideal\"].max():.3f}]')
# Check that normalization was applied (mean near 0, std near 1)
print(f'RSS per-feature mean: {d[\"rss\"].mean(axis=0)[:4]}')
print(f'RSS per-feature std: {d[\"rss\"].std(axis=0)[:4]}')
"
```

- [ ] **Step 4: Commit**

```bash
git add uwb_rti/data_generator.py
git commit -m "feat: add data generator with SLF images, noise, and RSS computation"
```

---

### Task 7: Deep Learning Models

**Files:**
- Create: `uwb_rti/models/mlp_model.py`
- Create: `uwb_rti/models/cfp_model.py`

- [ ] **Step 1: Write mlp_model.py**

```python
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 900),
        )

    def forward(self, x):
        return self.network(x)
```

- [ ] **Step 2: Write cfp_model.py**

```python
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class CFPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Shortcut path: Conv2d(5x5, 1 filter)
        self.shortcut = nn.Conv2d(1, 1, kernel_size=5, padding=2)

        # Residual path
        self.res_entry = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.res_bn_relu = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Post-merge layers
        self.merge_bn_relu = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_48 = nn.Conv2d(32, 48, kernel_size=1)
        self.skip_proj = nn.Conv2d(32, 48, kernel_size=1)
        self.dropout = nn.Dropout(0.3)
        self.conv_out = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x):
        # Shortcut path
        shortcut = self.shortcut(x)  # (B, 1, 30, 30)

        # Residual path
        res = self.res_entry(x)  # (B, 32, 30, 30)
        res = self.res_blocks(res)  # (B, 32, 30, 30)
        res = self.res_bn_relu(res)  # (B, 32, 30, 30)

        # Merge: shortcut (1 ch) broadcast-added to residual (32 ch).
        # The single shortcut feature is added identically to all 32 channels.
        merge = shortcut + res  # (B, 32, 30, 30)

        # Post-merge with residual skip around Conv2d(1x1, 48).
        # skip_proj projects merge (32 ch) to 48 ch to match conv_48 output.
        out = self.merge_bn_relu(merge)  # (B, 32, 30, 30)
        out = self.conv_48(out) + self.skip_proj(merge)  # (B, 48, 30, 30)
        out = self.dropout(out)
        out = self.conv_out(out)  # (B, 1, 30, 30)

        return out
```

- [ ] **Step 3: Verify models instantiate and forward pass works**

```bash
python -c "
import torch
from uwb_rti.models.mlp_model import MLPModel
from uwb_rti.models.cfp_model import CFPModel

mlp = MLPModel()
x = torch.randn(4, 16)
out = mlp(x)
print(f'MLP: input {x.shape} -> output {out.shape}')

cfp = CFPModel()
x = torch.randn(4, 1, 30, 30)
out = cfp(x)
print(f'CFP: input {x.shape} -> output {out.shape}')

print(f'MLP params: {sum(p.numel() for p in mlp.parameters()):,}')
print(f'CFP params: {sum(p.numel() for p in cfp.parameters()):,}')
"
```

Expected: MLP input (4,16) -> output (4,900). CFP input (4,1,30,30) -> output (4,1,30,30).

- [ ] **Step 4: Commit**

```bash
git add uwb_rti/models/mlp_model.py uwb_rti/models/cfp_model.py
git commit -m "feat: add MLP and CFP model architectures"
```

---

### Task 8: Training Script

**Files:**
- Create: `uwb_rti/train.py`

- [ ] **Step 1: Write train.py**

```python
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .config import (
    DEVICE, BATCH_SIZE, RANDOM_SEED, GRID_SIZE,
    MLP_LR, MLP_EPOCHS, MLP_PATIENCE,
    CFP_LR, CFP_MOMENTUM, CFP_EPOCHS,
    MLP_TRAIN_SIZE, MLP_VAL_SIZE,
    CFP_TRAIN_SIZE, CFP_VAL_SIZE,
)
from .models.mlp_model import MLPModel
from .models.cfp_model import CFPModel


def train_loop(model, train_loader, val_loader, criterion, optimizer,
               scheduler, epochs, patience, device):
    """Training loop with optional early stopping.

    Returns:
        history: dict with 'train_loss' and 'val_loss' lists.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                total_val_loss += loss.item() * batch_x.size(0)

        val_loss = total_val_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train: {train_loss:.6f} - Val: {val_loss:.6f} - LR: {lr:.2e}")

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_mlp(data_dir="data", checkpoint_dir="checkpoints"):
    """Train MLP model on MLP dataset."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data = np.load(os.path.join(data_dir, "mlp_data.npz"))
    rss = data["rss"]
    theta_ideal = data["theta_ideal"]

    X_train = torch.FloatTensor(rss[:MLP_TRAIN_SIZE])
    y_train = torch.FloatTensor(theta_ideal[:MLP_TRAIN_SIZE])
    X_val = torch.FloatTensor(rss[MLP_TRAIN_SIZE:MLP_TRAIN_SIZE + MLP_VAL_SIZE])
    y_val = torch.FloatTensor(theta_ideal[MLP_TRAIN_SIZE:MLP_TRAIN_SIZE + MLP_VAL_SIZE])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    model = MLPModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    print("Training MLP...")
    history = train_loop(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, MLP_EPOCHS, MLP_PATIENCE, DEVICE,
    )

    path = os.path.join(checkpoint_dir, "mlp_best.pt")
    torch.save(model.state_dict(), path)
    print(f"MLP model saved to {path}")

    return model, history


def generate_cfp_training_data(mlp_model, data_dir="data"):
    """Run trained MLP on CFP dataset to produce training pairs for CFP."""
    data = np.load(os.path.join(data_dir, "cfp_data.npz"))
    rss = data["rss"]
    theta_ideal = data["theta_ideal"]

    mlp_model.eval()
    n = rss.shape[0]
    mlp_output = np.zeros((n, 900))

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = torch.FloatTensor(rss[start:end]).to(DEVICE)
            mlp_output[start:end] = mlp_model(batch).cpu().numpy()

    mlp_images = mlp_output.reshape(-1, 1, GRID_SIZE, GRID_SIZE)
    ideal_images = theta_ideal.reshape(-1, 1, GRID_SIZE, GRID_SIZE)

    np.savez(
        os.path.join(data_dir, "cfp_training_pairs.npz"),
        mlp_images=mlp_images, ideal_images=ideal_images,
    )
    print(f"CFP training pairs saved ({mlp_images.shape[0]} samples)")

    return mlp_images, ideal_images


def train_cfp(data_dir="data", checkpoint_dir="checkpoints"):
    """Train CFP model on MLP output / ideal SLF pairs."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(RANDOM_SEED + 100)
    np.random.seed(RANDOM_SEED + 100)

    data = np.load(os.path.join(data_dir, "cfp_training_pairs.npz"))
    mlp_images = data["mlp_images"]
    ideal_images = data["ideal_images"]

    X_train = torch.FloatTensor(mlp_images[:CFP_TRAIN_SIZE])
    y_train = torch.FloatTensor(ideal_images[:CFP_TRAIN_SIZE])
    X_val = torch.FloatTensor(mlp_images[CFP_TRAIN_SIZE:CFP_TRAIN_SIZE + CFP_VAL_SIZE])
    y_val = torch.FloatTensor(ideal_images[CFP_TRAIN_SIZE:CFP_TRAIN_SIZE + CFP_VAL_SIZE])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    model = CFPModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=CFP_LR, momentum=CFP_MOMENTUM)

    print("Training CFP...")
    history = train_loop(
        model, train_loader, val_loader, criterion, optimizer,
        None, CFP_EPOCHS, None, DEVICE,
    )

    path = os.path.join(checkpoint_dir, "cfp_best.pt")
    torch.save(model.state_dict(), path)
    print(f"CFP model saved to {path}")

    return model, history
```

- [ ] **Step 2: Verify train.py imports correctly**

```bash
python -c "from uwb_rti.train import train_mlp, train_cfp, generate_cfp_training_data; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add uwb_rti/train.py
git commit -m "feat: add training script with MLP and CFP training loops"
```

---

### Task 9: Evaluation Module

**Files:**
- Create: `uwb_rti/evaluate.py`

- [ ] **Step 1: Write evaluate.py**

```python
import os
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from .config import (
    DEVICE, GRID_SIZE, NUM_LINKS, NUM_PIXELS,
    SCALING_C, RANDOM_SEED, BATCH_SIZE,
)
from .models.mlp_model import MLPModel
from .models.cfp_model import CFPModel


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
```

- [ ] **Step 2: Verify evaluate.py imports correctly**

```bash
python -c "from uwb_rti.evaluate import evaluate_on_test_set, evaluate_across_noise_levels; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add uwb_rti/evaluate.py
git commit -m "feat: add evaluation module with RMSE, SSIM, and multi-noise comparison"
```

---

### Task 10: Main Pipeline

**Files:**
- Create: `uwb_rti/main.py`

- [ ] **Step 1: Write main.py**

```python
import argparse
import os
import numpy as np
import torch

from .config import DEVICE, RANDOM_SEED, GRID_SIZE
from .forward_model import compute_weight_matrix
from .validate_model import run_all_validations
from .data_generator import generate_and_save_all_datasets
from .train import train_mlp, generate_cfp_training_data, train_cfp
from .evaluate import evaluate_on_test_set, evaluate_across_noise_levels
from .visualize import (
    plot_weight_matrix, plot_training_curves,
    plot_reconstruction_grid, plot_error_maps,
    plot_noise_comparison_table,
)
from .models.mlp_model import MLPModel
from .models.cfp_model import CFPModel


def load_models(checkpoint_dir="checkpoints"):
    """Load trained MLP and CFP models from checkpoints."""
    mlp_model = MLPModel().to(DEVICE)
    mlp_path = os.path.join(checkpoint_dir, "mlp_best.pt")
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=DEVICE, weights_only=True))
    mlp_model.eval()

    cfp_model = CFPModel().to(DEVICE)
    cfp_path = os.path.join(checkpoint_dir, "cfp_best.pt")
    cfp_model.load_state_dict(torch.load(cfp_path, map_location=DEVICE, weights_only=True))
    cfp_model.eval()

    return mlp_model, cfp_model


def stage_validate():
    """Stage 1: Validate forward model."""
    print("\n" + "=" * 60)
    print("STAGE: Forward Model Validation")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    W = compute_weight_matrix()
    plot_weight_matrix(W, [0, 5, 10, 15])
    run_all_validations()


def stage_generate():
    """Stage 2: Generate all datasets."""
    print("\n" + "=" * 60)
    print("STAGE: Data Generation")
    print("=" * 60)
    generate_and_save_all_datasets()


def stage_train_mlp():
    """Stage 3: Train MLP model."""
    print("\n" + "=" * 60)
    print("STAGE: MLP Training")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    mlp_model, mlp_history = train_mlp()
    plot_training_curves(mlp_history, "MLP Training", "figures/mlp_training.png")

    generate_cfp_training_data(mlp_model)


def stage_train_cfp():
    """Stage 4: Train CFP model."""
    print("\n" + "=" * 60)
    print("STAGE: CFP Training")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    cfp_model, cfp_history = train_cfp()
    plot_training_curves(cfp_history, "CFP Training", "figures/cfp_training.png")


def stage_evaluate():
    """Stage 5: Evaluate and visualize results."""
    print("\n" + "=" * 60)
    print("STAGE: Evaluation")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    mlp_model, cfp_model = load_models()

    results, mlp_out, cfp_out, theta_ideal = evaluate_on_test_set(mlp_model, cfp_model)

    plot_reconstruction_grid(
        theta_ideal, mlp_out, cfp_out, num_samples=5,
        save_path="figures/reconstruction_grid.png",
    )
    plot_error_maps(
        theta_ideal, mlp_out, cfp_out, num_samples=5,
        save_path="figures/error_maps.png",
    )

    noise_results = evaluate_across_noise_levels(mlp_model, cfp_model)
    plot_noise_comparison_table(noise_results)


def main():
    parser = argparse.ArgumentParser(description="UWB RTI Deep Learning Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "validate", "generate", "train_mlp", "train_cfp", "evaluate"],
        default="all",
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")

    stages = {
        "validate": stage_validate,
        "generate": stage_generate,
        "train_mlp": stage_train_mlp,
        "train_cfp": stage_train_cfp,
        "evaluate": stage_evaluate,
    }

    if args.stage == "all":
        for name, fn in stages.items():
            fn()
    else:
        stages[args.stage]()

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify main.py imports correctly**

```bash
python -c "from uwb_rti.main import main; print('OK')"
```

- [ ] **Step 3: Run the full pipeline**

```bash
python -m uwb_rti.main --stage all
```

Expected: Runs all stages sequentially — validation, data generation, MLP training, CFP training, evaluation. All plots saved to `figures/`, checkpoints to `checkpoints/`.

- [ ] **Step 4: Commit**

```bash
git add uwb_rti/main.py
git commit -m "feat: add main pipeline with CLI stage selection"
```

- [ ] **Step 5: Add data and output directories to .gitignore**

Create `.gitignore`:
```
data/
checkpoints/
figures/
__pycache__/
*.pyc
```

```bash
git add .gitignore
git commit -m "chore: add .gitignore for data, checkpoints, figures, and cache"
```
