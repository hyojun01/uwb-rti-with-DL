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
MLP_EPOCHS = 300
MLP_PATIENCE = 30
CFP_LR = 0.001
CFP_MOMENTUM = 0.9
CFP_EPOCHS = 300
MLP_ENSEMBLE_SIZE = 3
GRAD_CLIP_NORM = 1.0
L1_WEIGHT = 0.2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
