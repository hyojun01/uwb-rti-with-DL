# UWB RTI Deep Learning System — Design Document

## Overview

Implement the UWB RTI system exactly as specified in `uwb_rti_spec.md`. Sequential build with validation gates at each stage. PyTorch for deep learning, NumPy for the forward model, matplotlib for visualization. Local CUDA GPU for training. Hardware integration with DW3000 via serial/UART planned for later — code must have a clean data ingestion boundary.

## Implementation Approach

**Sequential with validation gates (Approach A):**

1. `config.py` + `forward_model.py` → validate with weight matrix visualization
2. `validate_model.py` → run both validation plots to confirm math before generating data
3. `data_generator.py` → generate datasets, spot-check samples
4. MLP model → train → evaluate
5. CFP model → train on MLP outputs → evaluate
6. Final visualizations and comparison

Each stage is verified before proceeding to the next.

## Module Designs

### `config.py` — Constants and Parameters

All physical, geometric, and training constants in one place:

- **Physical:** `C_LIGHT = 3e8`, `F_CENTER = 7.9872e9`, `BANDWIDTH = 499.2e6`, `LAMBDA = C_LIGHT / F_CENTER`
- **Geometry:** `AREA_SIZE = 3.0`, `GRID_SIZE = 30`, `NUM_PIXELS = 900`, `PIXEL_SIZE = AREA_SIZE / GRID_SIZE`
- **Nodes:** `TX_POSITIONS` (4 tags at y=0), `RX_POSITIONS` (4 anchors at y=3), `NUM_LINKS = 16`
- **Weight model:** `BETA_MIN = LAMBDA / 20`, `SCALING_C = 2`. `BETA_MAX` computed per link via `sqrt(lambda * d_{n,m} / 4)`.
- **Parameter ranges:** bias U(90,100), alpha U(0.9,1.0), noise sigma U(0.3,3.0), SLF noise sigma U(0.01,0.05), kappa=0.21
- **Training:** dataset sizes (40k MLP, 20k CFP, 1k test), batch size, learning rates, epochs, patience — all per spec
- **Device:** auto-detect CUDA via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- **Reproducibility:** `RANDOM_SEED` constant, set in NumPy and PyTorch at data generation and training entry points

Pixel center positions: `x_k = (c + 0.5) * (3.0 / 30)`, `y_k = (r + 0.5) * (3.0 / 30)`.

Node positions and pixel grid stored as NumPy arrays. Everything else as plain Python constants.

### `forward_model.py` — Weight Matrix and RSS Generation

**`compute_weight_matrix()`** — Returns W (16x900 NumPy array):
- Iterates over all 16 TX-RX pairs and 900 pixels
- For each (link, pixel): compute d1, d2 (distances to TX/RX), semi-major axis `a = (d1+d2)/2`, half-foci distance `c_half = d_{n,m}/2`, semi-minor axis `beta = sqrt(a^2 - c_half^2)`
- Inverse area elliptical model (Hamilton et al. 2014):
  - If `beta >= beta_max`: weight = 0 (outside Fresnel zone)
  - If `beta <= beta_min`: weight = `1 / (pi * a * beta_min)` (clamped)
  - Otherwise: weight = `1 / (pi * a * beta)`
- `beta_max = sqrt(lambda * d_{n,m} / 4)` computed per link

**`compute_rss(theta, b, alpha, sigma_eps, W, d)`** — Returns y (16,):
- `d` vector: `20 * log10(D)` for each link's TX-RX distance
- `epsilon = N(0, sigma_eps^2)` drawn per call
- `y = b - SCALING_C * W @ theta - alpha * d + epsilon`

**`compute_distance_vector()`** — Returns d (16,) with log-distances for all links.

All vectorized with NumPy. Weight matrix computed once and reused.

### `data_generator.py` — Training Data Generation

**`generate_slf_image()`:**
- Creates theta* (900,) by placing 1-3 random objects on the 30x30 grid
- 10 object types from spec Section 2.7 (standing person, walking person, table, chair, cabinet, wall segment, multiple objects, empty room, L/T-shaped, circular pillar)
- Each type has its specified size range and attenuation U(low, high)
- Objects placed randomly with 0.2m margin from edges
- Rectangular objects: set pixels within rectangle to sampled attenuation
- Circular objects: set pixels within radius to sampled attenuation
- L/T-shaped: compose from two rectangles

**`generate_slf_noise()`:**
- Covariance matrix C (900x900): `C[k,l] = sigma_theta^2 * exp(-D_kl / kappa)`
- D_kl is Euclidean distance between pixel centers k and l
- Sample from `N(0, C)` using Cholesky decomposition
- Pre-compute `L = cholesky(C_base)` where `C_base[k,l] = exp(-D_kl / kappa)` (once). For each sample, generate noise as `theta_tilde = sigma_theta * L @ z` where `z ~ N(0, I)`

**`generate_dataset(num_samples)`:**
- For each sample: generate theta*, sample sigma_theta ~ U(0.01, 0.05), compute theta = theta* + noise, sample b/alpha/sigma_eps, compute RSS
- Returns: RSS vectors (N, 16), noisy SLF images (N, 900), ideal SLF images (N, 900)
- Total generated samples: 61,000 (40k MLP + 20k CFP + 1k separate test set)
- Saves to disk as `.npz` files

**Normalization:**
- Compute mean/std of RSS from MLP training set (36k samples)
- Store stats and apply to all splits

### `models/mlp_model.py` — MLP Architecture

PyTorch `nn.Module`:
```
Input(16)
-> FC(256) -> BatchNorm1d -> ReLU -> Dropout(0.3)
-> FC(512) -> BatchNorm1d -> ReLU -> Dropout(0.3)
-> FC(1024) -> BatchNorm1d -> ReLU -> Dropout(0.3)
-> FC(2048) -> BatchNorm1d -> ReLU -> Dropout(0.2)
-> FC(900) -> Linear activation (no activation)
```

### `models/cfp_model.py` — CFP CNN Architecture

PyTorch `nn.Module`. Input: (batch, 1, 30, 30).

- **Shortcut path:** Conv2d(5x5, 1 filter, padding=2)
- **Residual path:**
  - Conv2d(3x3, 32 filters, padding=1) -> BN -> ReLU
  - 4 residual blocks, each:
    - Conv2d(3x3, 16 filters, padding=1) -> BN -> ReLU
    - Conv2d(1x1, 32 filters)
    - Add input of block (residual connection)
  - BN -> ReLU
- **Merge:** Add shortcut + residual path
- BN -> ReLU -> Conv2d(1x1, 48 filters) -> Add the merge output (residual skip around the 1x1 48-filter conv) -> Dropout(0.3) -> Conv2d(1x1, 1 filter) -> Output (batch, 1, 30, 30)

### `train.py` — Training Script

**MLP training:**
- `TensorDataset` + `DataLoader`, 36k train / 4k val
- `MSELoss`, `Adam(lr=1e-3)`, `ReduceLROnPlateau`, early stopping patience=20
- Saves best checkpoint, logs train/val loss per epoch

**CFP training:**
- Run trained MLP on 20k CFP split to produce reconstructed images
- (MLP output, ideal SLF) pairs, 18k train / 2k val
- `MSELoss`, `SGD(lr=0.001, momentum=0.9)`, up to 200 epochs
- Saves best checkpoint

**Shared:** Training loop function reused for both models. All tensors on CUDA. Checkpoints as `.pt` files.

### `evaluate.py` — Evaluation and Metrics

- Runs MLP and MLP+CFP pipeline on 1k test set
- Per-sample RMSE: `sqrt(mean((theta_pred - theta_true)^2))`
- Per-sample SSIM via `skimage.metrics.structural_similarity` on 30x30 images
- Prints comparison table with average RMSE and SSIM
- **Multi-noise-level evaluation:** Generate or evaluate test sets at several fixed noise levels (e.g., sigma_eps in {0.5, 1.0, 2.0, 3.0}) and produce a quantitative comparison table of RMSE and SSIM for MLP vs MLP+CFP across these levels

### `validate_model.py` — Mathematical Model Validation

- **Validation 1 (RSS vs Distance):** Single TX-RX pair, no objects, b=95, alpha=0.95, D from 0.5m to 5m. Plots `y = b - alpha * 20 * log10(D)`.
- **Validation 2 (Human Crossing):** TX at (0,0), RX at (0,3). Person (0.4mx0.4m, theta*=0.7) moves from x=-1 to x=4 at y=1.5. Plots RSS vs x-position showing LOS dip.

### `visualize.py` — All Visualization Functions

- Weight matrix heatmaps (selected links, reshaped to 30x30)
- Training loss curves (train/val vs epoch)
- Reconstruction grid: ground truth | MLP output | MLP+CFP output
- Error maps: |predicted - ground truth| per method
- Quantitative comparison table across noise levels (from evaluate.py results)
- All plots saved as PNG via matplotlib

### `main.py` — Main Execution Pipeline

- Orchestrates full pipeline in spec priority order
- CLI args for stages (`--stage validate`, `--stage train_mlp`, `--stage all`)
- Each stage independently callable

## Dependencies

- `torch` (+ CUDA)
- `numpy`
- `matplotlib`
- `scikit-image` (for SSIM)

## File Structure

```
uwb_rti/
├── config.py
├── forward_model.py
├── data_generator.py
├── models/
│   ├── mlp_model.py
│   └── cfp_model.py
├── train.py
├── evaluate.py
├── validate_model.py
├── visualize.py
└── main.py
```

## Hardware Integration Notes

The serial/UART interface to DW3000 will be integrated later. The key boundary is the RSS vector input to the MLP model — simulated data produces a (16,) normalized vector, and real hardware will need to produce the same shape. The normalization stats (mean/std) will need recalibration on real data.
