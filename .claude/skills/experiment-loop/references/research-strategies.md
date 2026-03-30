# Research Strategies for UWB RTI Model Optimization

## Understanding the problem

UWB RTI is an inverse problem: 16 RSS measurements → 900-pixel SLF image.
The weight matrix W (16×900) has rank at most 16, so 884 dimensions of θ are in the null space.
The DL model must learn strong structural priors from training data to fill in null space information.

## Priority order for experiments

### Phase 1: Baseline and low-hanging fruit (experiments 0-5)

1. **Baseline** — Always run first, unmodified
2. **Learning rate sweep** — Try 3e-4, 1e-3, 3e-3 for MLP
3. **Dropout tuning** — Try 0.1, 0.2, 0.3, 0.5 for MLP
4. **Batch size** — Try 128, 256, 512
5. **Activation function** — GELU, SiLU, LeakyReLU vs current ReLU

### Phase 2: Architecture changes (experiments 6-15)

6. **MLP width** — Wider (2x) or narrower (0.5x) hidden layers
7. **MLP depth** — Add or remove hidden layers
8. **Skip connections in MLP** — Concatenate input to intermediate layers
9. **Residual MLP** — Add skip connections between adjacent hidden layers
10. **CFP depth** — More or fewer residual blocks
11. **CFP width** — More or fewer channels
12. **CFP kernel sizes** — Try 7x7 shortcut instead of 5x5

### Phase 3: Training strategy (experiments 16-25)

13. **Loss function** — Huber loss, L1 loss, combined L1+MSE
14. **Optimizer** — AdamW with weight decay, SGD with momentum for MLP
15. **LR scheduler** — CosineAnnealingLR, OneCycleLR, WarmupCosine
16. **Gradient clipping** — max_norm=1.0
17. **Label smoothing / noise augmentation** — Add noise to SLF targets during training
18. **Mixed precision** — torch.amp.autocast for faster training

### Phase 4: Advanced architecture (experiments 26+)

19. **Attention in MLP** — Self-attention on intermediate features
20. **Multi-scale CFP** — Dilated convolutions for larger receptive field
21. **U-Net style CFP** — Encoder-decoder with skip connections
22. **Separate encoders** — Different networks for different link subsets
23. **Physics-informed loss** — Add W·θ_pred ≈ W·θ_true as auxiliary loss
24. **Ensemble** — Average predictions from multiple trained models

## Key principles

### What typically helps in underdetermined inverse problems:
- Strong regularization (moderate dropout, weight decay)
- Smooth output (loss functions that penalize sharp edges)
- Physics-aware constraints
- Residual learning (predict the delta from a simple initial estimate)

### What typically hurts:
- Too much model capacity without regularization (overfits to training noise)
- Very deep networks without skip connections (gradient issues)
- Too aggressive LR (training instability with only 16 input dims)
- Ignoring the 2-stage dependency (CFP quality depends on MLP quality)

### Understanding the MLP→CFP interaction:
- MLP errors are **systematic** — similar inputs produce similar artifacts
- CFP learns to correct these systematic artifacts
- If MLP architecture changes significantly, CFP must be retrained
- A slightly worse MLP with more consistent errors may lead to better MLP+CFP

## Metric interpretation

- `test_rmse ≈ 0.10`: Good reconstruction quality
- `test_rmse ≈ 0.05`: Excellent (near pixel-perfect for non-zero regions)
- `test_ssim ≈ 0.80`: Good structural similarity
- `test_ssim ≈ 0.90`: Excellent

The CFP metrics (cfp_test_rmse, cfp_test_ssim) should always be better than MLP-only metrics. If not, the CFP is hurting rather than helping — investigate.

## Combining successful changes

After finding individual improvements, try combining them:
- If GELU helped (+) and wider layers helped (+), try GELU + wider layers
- If two changes each helped marginally, their combination may help more
- But watch for interactions: dropout + wider layers may need retuning
