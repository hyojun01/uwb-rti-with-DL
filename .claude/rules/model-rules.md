---
globs: ["uwb_rti/models/*.py"]
---

# Model architecture rules

## MLP model constraints

- Input dimension must remain 16 (RSS vector size, fixed by physics)
- Output dimension must remain 900 (30x30 SLF grid, fixed by physics)
- Must use `nn.Module` with a `forward(self, x)` method returning (batch, 900)
- Do not add external data dependencies (no loading files in forward pass)
- Keep the model single-GPU compatible (no distributed training)

## EnsembleMLPModel constraints

- Must wrap members in `nn.ModuleList` (so parameters are registered for save/load)
- `forward()` must accept the same input as a single MLPModel and return the same shape
- All members must have output dim 900 — architecture variations between members are allowed but output shape must match
- Aggregation method (mean, weighted mean, etc.) is modifiable but must produce (batch, 900)
- `torch.save(ensemble.state_dict(), path)` must work for checkpointing

## CFP model constraints

- Input shape must remain (batch, 1, 30, 30) — single-channel 30x30 image
- Output shape must remain (batch, 1, 30, 30)
- Must use `nn.Module` with a `forward(self, x)` method
- All convolutions must use `padding` to maintain spatial dimensions (no size changes)
- Do not use transposed convolutions or upsampling (input and output are same size)

## General

- Use only PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
- No external model libraries (no timm, no transformers, no einops)
- Keep parameter count reasonable — watch for OOM on single GPU
- Test the model instantiation: `model = MLPModel()` or `model = CFPModel()` must not error
- `EnsembleMLPModel([MLPModel(), MLPModel()])` must also not error
