# Deep Learning-Based IEEE 802.15.4z HRP UWB Radio Tomographic Imaging (RTI)

## Project Overview

Implement a complete UWB RTI system: mathematical forward model, training data generation, deep learning models (MLP, CFP), training, evaluation, and visualization. All code in Python.

---

## 1. Experimental Setup

| Parameter | Value |
|---|---|
| UWB Module | Qorvo DW3000 |
| Channel | CH9 (center freq ~7.9872 GHz, bandwidth ~499.2 MHz) |
| Standard | IEEE 802.15.4z HRP UWB |
| Imaging Area | 3 m × 3 m |
| Image Resolution | 30 × 30 pixels (K = 900) |
| Number of Tags (TX) | 4 |
| Number of Anchors (RX) | 4 |
| Total Links | N = 16 (each TX to each RX) |

### Node Positions

Tags (TX) are placed along the bottom edge (y = 0):
- TX0: (0, 0), TX1: (1, 0), TX2: (2, 0), TX3: (3, 0)

Anchors (RX) are placed along the top edge (y = 3):
- RX0: (0, 3), RX1: (1, 3), RX2: (2, 3), RX3: (3, 3)

Each pixel center position: for pixel index k with row r = k // 30, col c = k % 30:
- x_k = (c + 0.5) * (3.0 / 30)
- y_k = (r + 0.5) * (3.0 / 30)

---

## 2. Mathematical Forward Model

### 2.1 RSS Measurement Model

Reference: Wu et al. (2020), Eq. (1).

For transmitter at x' and receiver at x'':

```
y = b - s - α * d + ε
```

Where:
- `y`: RSS measurement in dB
- `b`: bias term (aggregates transmitter power, receiver sensitivity, antenna gains)
- `s`: shadowing component (signal attenuation due to objects)
- `α`: free space path loss exponent
- `d = 20 * log10(D)`: log-distance, D is distance in meters between TX and RX
- `ε`: additive measurement noise

### 2.2 Shadowing Component

Reference: Wu et al. (2020), Eq. (2).

```
s(θ, x', x'') = c * Σ_{k=1}^{K} w(x', x'', x_k) * θ_k = c * w^T * θ
```

Where:
- `θ ∈ R^K` (K=900): discrete SLF vector
- `w ∈ R^K`: weight vector for a given TX-RX pair
- `x_k`: center position of pixel k
- `c = 2`: scaling constant

### 2.3 Full Forward Model

Reference: Wu et al. (2020), Eq. (5).

For all 16 links (N=16), each link has 1 measurement (static nodes, n=1):

```
y = Z*b - c*W*θ - α*d + ε
```

Where:
- `y ∈ R^16`: full RSS measurement vector
- `Z ∈ R^{16×16}`: identity matrix (since n=1, Z = I_16)
- `b ∈ R^16`: bias vector for each link
- `W ∈ R^{16×900}`: weight matrix (each row is the weight vector for one link)
- `θ ∈ R^900`: SLF image vector
- `d ∈ R^16`: log-distance vector
- `ε ∈ R^16`: noise vector

### 2.4 Weight Model: Inverse Area Elliptical Model

Reference: Hamilton et al. (2014), Eqs. (19)-(20).

For a pixel at position s, with transmitter at s_n and receiver at s_m:

```
b(s_n, s_m, s) = 1 / (π * (d_{n,m}/2) * β(s))    if β_min < β(s) < β_max
               = 1 / (π * (d_{n,m}/2) * β_min)     if β(s) ≤ β_min
               = 0                                    if β(s) ≥ β_max
```

Where:
- `d_{n,m}`: distance between TX n and RX m
- `β(s)`: semi-minor axis of the smallest ellipse containing point s with TX and RX as foci

Computing β(s):
Given TX at position p1, RX at position p2, and pixel center at position p:
- `d1 = ||p - p1||` (distance from pixel to TX)
- `d2 = ||p - p2||` (distance from pixel to RX)
- `a = (d1 + d2) / 2` (semi-major axis)
- `c_half = d_{n,m} / 2` (half the distance between foci)
- `β(s) = sqrt(a^2 - c_half^2)` (semi-minor axis)

Bounds:
- `β_min → 0` (set to a small positive value, e.g., λ/20 where λ = c_light / f_center)
- `β_max`: semi-minor axis of the first Fresnel zone ellipse

First Fresnel zone semi-minor axis:
```
β_max = sqrt(λ * d_{n,m} / 4)
```
where λ = 3e8 / 7.9872e9 ≈ 0.03755 m

The ellipse area with semi-major axis a and semi-minor axis β:
```
Area = π * a * β ≈ π * (d_{n,m}/2) * β    (approximation when β << a)
```

More precisely, use the actual formula:
```
weight(s) = 1 / (π * a * β(s))
```

with clamping at β_min and cutoff at β_max.

### 2.5 Parameter Ranges

Reference: Wu et al. (2020), Section IV-A.

| Parameter | Distribution/Value |
|---|---|
| Bias b_i | U(90, 100) for each link |
| Path loss exponent α | U(0.9, 1.0) |
| Noise ε | N(0, σ_ε²·I), σ_ε ~ U(0.3, 3.0) |
| SLF noise σ_θ | U(0.01, 0.05) |
| SLF spatial correlation κ | 0.21 m |
| Scaling constant c | 2 |

### 2.6 SLF Image Definition

Reference: Wu et al. (2020), Section IV-A.

```
θ = θ* + θ̃
```

- `θ*`: ideal SLF image (ground truth)
  - Free space: θ*_k = 0
  - Object region: θ*_k ~ U(0.3, 1.0)
- `θ̃`: zero-mean spatially correlated Gaussian noise
  - Covariance: C_θ(k,l) = σ_θ² * exp(-D_kl / κ)
  - D_kl: distance between pixel k and pixel l centers

### 2.7 SLF Target Types (Indoor Environment)

Generate diverse training samples representing indoor scenarios:

1. **Person (standing)**: rectangular region ~0.4m × 0.4m, attenuation U(0.5, 1.0)
2. **Person (walking)**: rectangular region ~0.3m × 0.5m at various positions
3. **Table/Desk**: rectangular region ~0.8m × 0.6m, attenuation U(0.3, 0.6)
4. **Chair**: rectangular region ~0.4m × 0.4m, attenuation U(0.3, 0.5)
5. **Cabinet/Shelf**: rectangular region ~0.5m × 0.3m, attenuation U(0.5, 0.8)
6. **Wall segment**: thin rectangular region ~0.1m × 1.0m, attenuation U(0.6, 1.0)
7. **Multiple objects**: combination of 2-3 objects
8. **Empty room**: all zeros (no objects)
9. **L-shaped / T-shaped objects**: composite shapes
10. **Circular objects** (pillar): radius ~0.2-0.3m

Randomly place 1-3 objects per sample. Ensure objects stay within the 3m × 3m area with some margin (0.2m from edges).

---

## 3. Data Generation Pipeline

### 3.1 Steps

1. Define pixel grid (30×30) with center positions
2. Compute distance vector `d` for all 16 links
3. Compute weight matrix `W` (16×900) using Inverse Area Elliptical Model
4. For each training sample:
   a. Generate random ideal SLF image θ* (place random objects)
   b. Generate SLF noise θ̃ from spatially correlated Gaussian
   c. θ = θ* + θ̃
   d. Sample b, α, σ_ε
   e. Compute y = Z*b - c*W*θ - α*d + ε

### 3.2 Dataset Size

- Total: 60,000 samples
- MLP training: 40,000 (36,000 train + 4,000 validation)
- CFP training: 20,000 (18,000 train + 2,000 validation)
- Test set: separate 1,000 samples

### 3.3 Input Normalization

- RSS vector y: normalize to zero mean, unit standard deviation (compute stats from training set)

---

## 4. Deep Learning Models

### 4.1 MLP Model (for RTI)

Input: 16-dimensional RSS vector
Output: 900-dimensional SLF vector (reshaped to 30×30 image)

Architecture (start with this, iterate based on performance):
```
Input(16)
→ FC(256) → BatchNorm → ReLU → Dropout(0.3)
→ FC(512) → BatchNorm → ReLU → Dropout(0.3)
→ FC(1024) → BatchNorm → ReLU → Dropout(0.3)
→ FC(2048) → BatchNorm → ReLU → Dropout(0.2)
→ FC(900) → Linear activation
```

Loss: MSE
Optimizer: Adam, lr=1e-3 with ReduceLROnPlateau
Epochs: up to 200 with early stopping (patience=20)

### 4.2 CFP Model (CNN for Post-processing)

Reference: Wu et al. (2020), Fig. 3, Fig. 5(c).

Input: 30×30×1 (SLF image from MLP output, reshaped from 900-dim vector)
Output: 30×30×1 (enhanced SLF image)

Architecture:
- **Conv shortcut path**: Conv2d(5×5, 1 filter, same padding)
- **Residual path**:
  - Conv2d(3×3, 32 filters, same padding) → BN → ReLU
  - Conv2d(1×1, n filters) → residual connection
  - Repeat residual blocks 3-4 times
  - Structure per residual block:
    - Conv2d(3×3, 16 filters, same padding) → BN → ReLU
    - Conv2d(1×1, 32 filters, same padding)
    - Add (residual connection)
- **Merge**: Add shortcut and residual outputs
- BN → ReLU
- Conv2d(1×1, 48 filters) → Addition
- Dropout(0.3)
- Conv2d(1×1, 1 filter) → Output

Loss: MSE
Optimizer: SGD with momentum 0.9, initial lr=0.001
Epochs: up to 200

### 4.3 Training Notes

- MLP and CFP must use **different** data subsets
- First train MLP on 40,000 samples
- Then run trained MLP on remaining 20,000 samples to get reconstructed images
- Train CFP using (reconstructed image, ideal SLF image) pairs from these 20,000 samples
- Evaluate on held-out test set of 1,000 samples

---

## 5. Model Validation (Mathematical Model)

### 5.1 Validation 1: RSS vs Distance (No Shadowing)

Setup: 1 TX, 1 RX. No objects (s = 0). Vary distance D from 0.5m to 5m.

Theoretical RSS:
```
y = b - α * 20 * log10(D)
```

Use typical values: b = 95 dB, α = 0.95.

Plot: theoretical RSS vs distance curve. This validates the path loss model.

Expected behavior: RSS decreases logarithmically with distance.

Real measurement reference: With DW3000 at CH9, the RSS (reported as CIR power or first path power) should follow a similar log-distance decay.

### 5.2 Validation 2: RSS Change During Human Crossing

Setup: 1 TX at (0, 0), 1 RX at (0, 3). Fixed distance D = 3m. A person crosses perpendicular to the TX-RX line at the midpoint.

Model the person as a rectangular object (0.4m × 0.4m, θ* = 0.7) moving from x = -1m to x = 4m at y = 1.5m.

For each person position:
1. Construct SLF image with person at current position
2. Compute shadowing: s = c * w^T * θ
3. Compute RSS: y = b - s - α * d

Plot: RSS vs person's x-position. Expected: RSS dip when person crosses the LOS path, with maximum attenuation at x = 0 (directly on LOS).

---

## 6. Evaluation Metrics

- **RMSE**: sqrt(mean((θ_pred - θ_true)²)) per sample, then average
- **SSIM**: structural similarity between predicted and true 30×30 images
- **Visual comparison**: plot predicted vs ground truth for representative test samples

---

## 7. Visualization Requirements

1. **Weight matrix visualization**: Show weight vector reshaped to 30×30 for a few representative links
2. **Model validation plots**: RSS vs distance, RSS vs person position
3. **Training curves**: loss vs epoch for all models
4. **Reconstruction comparison**: grid showing ground truth, MLP output, MLP+CFP output
5. **Quantitative comparison table**: RMSE and SSIM for MLP, MLP+CFP across different noise levels
6. **Error maps**: |predicted - ground truth| for each method

---

## 8. File Structure

```
uwb_rti/
├── config.py              # All constants and parameters
├── forward_model.py       # Weight matrix, RSS generation
├── data_generator.py      # Training data generation
├── models/
│   ├── mlp_model.py       # MLP architecture
│   └── cfp_model.py       # CFP CNN architecture
├── train.py               # Training script for all models
├── evaluate.py            # Evaluation and metrics
├── validate_model.py      # Mathematical model validation (Section 5)
├── visualize.py           # All visualization functions
└── main.py                # Main execution pipeline
```

---

## 9. Implementation Priority

1. `config.py` and `forward_model.py` (weight matrix computation)
2. `validate_model.py` (verify forward model correctness)
3. `data_generator.py` (generate training data)
4. MLP model → train → evaluate
5. CFP model → train with MLP outputs → evaluate
6. Comparative visualization and analysis
