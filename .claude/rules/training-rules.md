---
globs: ["uwb_rti/train.py"]
---

# Training code rules

## Pipeline invariants

- Must call `train_mlp()` before `generate_cfp_training_data()` before `train_cfp()`
- `train_mlp()` must return `(model, history)` where history has `train_loss` and `val_loss` lists
- `generate_cfp_training_data()` must take `(mlp_model, data_dir)` and save `cfp_training_pairs.npz`
- `train_cfp()` must return `(model, history)` with same format
- The `train_loop()` function signature and return format must not change

## Data handling

- MLP training uses `mlp_data.npz` with keys `rss` and `theta_ideal`
- CFP training uses `cfp_training_pairs.npz` with keys `mlp_images` and `ideal_images`
- MLP input shape: (batch, 16), target: (batch, 900)
- CFP input shape: (batch, 1, 30, 30), target: (batch, 1, 30, 30)
- Do not change the data loading logic or file format

## Safe to modify

- Optimizer choice and parameters (Adam, AdamW, SGD, etc.)
- Learning rate scheduler choice and parameters
- Loss function (MSE, Huber, L1, combinations)
- Gradient clipping
- Mixed precision training
- Early stopping patience
- Training loop internals (gradient accumulation, logging frequency)
