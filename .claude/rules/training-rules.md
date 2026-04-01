---
globs: ["uwb_rti/train.py"]
---

# Training code rules

## Pipeline invariants

- Must call `train_mlp()` before `generate_cfp_training_data()` before `train_cfp()`
- `train_mlp()` must return `(model, history)` where:
  - `model` is an `nn.Module` (either `EnsembleMLPModel` or `MLPModel`) that accepts (batch, 16) and returns (batch, 900)
  - `history` has `train_loss` and `val_loss` lists (at minimum the best member's history)
- `generate_cfp_training_data()` must take `(mlp_model, data_dir)` and save `cfp_training_pairs.npz`
- `train_cfp()` must return `(model, history)` with same format
- The `train_loop()` function signature and return format must not change

## Ensemble training constraints

- Each ensemble member must be trained independently (no shared parameters)
- The ensemble must be wrapped in `EnsembleMLPModel` before being returned
- `EnsembleMLPModel.forward()` must return the averaged predictions of all members
- Member diversity can come from: different seeds, data subsampling, hyperparameter variation, or architecture variation
- All members must have the same input (16) and output (900) dimensions
- Training time scales linearly with ensemble size — verify experiments fit within 60-minute budget

## Data handling

- MLP training uses `mlp_data.npz` with keys `rss` and `theta_ideal`
- CFP training uses `cfp_training_pairs.npz` with keys `mlp_images` and `ideal_images`
- MLP input shape: (batch, 16), target: (batch, 900)
- CFP input shape: (batch, 1, 30, 30), target: (batch, 1, 30, 30)
- Data augmentation in `data_generator.py` is allowed but must not change the data format
- Bagging (per-member random subsampling) modifies the DataLoader, not the saved data files

## Safe to modify

- Optimizer choice and parameters (Adam, AdamW, SGD, etc.)
- Learning rate scheduler choice and parameters
- Loss function (MSE, Huber, L1, combinations)
- Gradient clipping
- Mixed precision training
- Early stopping patience
- Training loop internals (gradient accumulation, logging frequency)
- Ensemble member training strategy (seed diversity, bagging, hyperparameter diversity)
- Ensemble aggregation method (mean, weighted mean, median)
