---
globs: ["uwb_rti/config.py"]
---

# Config modification rules

## NEVER modify these (physics/experiment constants)

- `C_LIGHT`, `F_CENTER`, `BANDWIDTH`, `LAMBDA` — physical constants
- `AREA_SIZE`, `GRID_SIZE`, `NUM_PIXELS`, `PIXEL_SIZE` — geometry
- `TX_POSITIONS`, `RX_POSITIONS`, `NUM_TX`, `NUM_RX`, `NUM_LINKS` — node setup
- `PIXEL_CENTERS` — derived from geometry
- `BETA_MIN`, `SCALING_C` — weight model parameters
- `BIAS_RANGE`, `ALPHA_RANGE`, `NOISE_SIGMA_RANGE`, `SLF_NOISE_SIGMA_RANGE`, `KAPPA` — simulation params
- `MLP_DATASET_SIZE`, `CFP_DATASET_SIZE`, `TEST_DATASET_SIZE` — dataset sizes
- `MLP_TRAIN_SIZE`, `MLP_VAL_SIZE`, `CFP_TRAIN_SIZE`, `CFP_VAL_SIZE` — split sizes
- `RANDOM_SEED` — reproducibility
- `DEVICE` — hardware detection

## CAN modify these (training hyperparameters)

- `BATCH_SIZE` — values: 64, 128, 256, 512
- `MLP_LR` — range: 1e-5 to 1e-2
- `MLP_EPOCHS` — range: 50 to 500
- `MLP_PATIENCE` — range: 5 to 50
- `CFP_LR` — range: 1e-5 to 1e-2
- `CFP_MOMENTUM` — range: 0.8 to 0.99
- `CFP_EPOCHS` — range: 50 to 500

## CAN add new hyperparameters

- New constants for model architecture (e.g., `MLP_HIDDEN_DIMS`, `CFP_NUM_BLOCKS`)
- New training parameters (e.g., `WEIGHT_DECAY`, `GRAD_CLIP_NORM`)
- Always add at the end of the "Training hyperparameters" section
