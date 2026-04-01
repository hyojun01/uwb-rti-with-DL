# Research Strategies for UWB RTI Ensemble Optimization

## Understanding the problem

UWB RTI is an inverse problem: 16 RSS measurements -> 900-pixel SLF image.
The weight matrix W (16x900) has rank at most 16, so 884 dimensions of theta are in the null space.
The DL model must learn strong structural priors from training data to fill in null space information.

**Why ensemble helps this problem specifically:**
The 16->900 mapping is massively underdetermined. A single model commits to one solution in the null space. Multiple models, initialized differently, converge to different solutions. Averaging these solutions tends to cancel out individual model artifacts while preserving the signal that all models agree on.

## Previous session findings (43 experiments, single MLP)

**What worked:** wider MLP [512->4096], Adam for CFP, OneCycleLR, MSE+L1 loss
**What failed:** GELU/SiLU activations, dropout reduction, sigmoid output, wider CFP, deeper/tapered MLP, weight decay

These findings carry over to ensemble members -- each member should use the proven configuration as its base.

## Priority order for ensemble experiments

### Phase 1: Establish ensemble baseline (experiments E01-E02)

1. **Ensemble baseline** -- Run current code (N=3, seed-only diversity) to establish metrics
2. **Ensemble size sweep** -- Try N=2, 3, 5 to find the sweet spot
   - N=1 is equivalent to single model (sanity check)
   - N=2 is minimum ensemble; fastest training
   - N=5 adds more diversity but 5x training time
   - Beyond N=5 is unlikely to justify the time cost

### Phase 2: Diversity methods (experiments E03-E06)

3. **Bagging** -- Each member trains on random 80% of data
   - Implement via `torch.utils.data.Subset` with random indices per member
   - Classic ensemble technique; increases decorrelation between members
   - Risk: 20% less data per member may hurt individual quality

4. **Dropout diversity** -- Each member uses slightly different dropout rates
   - e.g., member 0: 0.25, member 1: 0.30, member 2: 0.35
   - Different regularization = different learned features
   - Easy to implement: pass dropout as parameter to MLPModel constructor

5. **Learning rate diversity** -- Each member uses slightly different max_lr
   - e.g., member 0: 2e-3, member 1: 3e-3, member 2: 4e-3
   - Different convergence speeds = different minima

6. **Architecture diversity** -- Vary hidden layer widths per member
   - e.g., member 0: [512,1024,2048,4096], member 1: [256,512,1024,2048], member 2: [1024,2048,4096,4096]
   - Highest diversity but requires modifying MLPModel to accept config
   - Risk: weaker members may drag down the average

### Phase 3: Efficient ensemble methods (experiments E07-E09)

7. **Weighted ensemble** -- Learn member weights on validation set
   - After training all members, optimize scalar weights w_i to minimize val_loss
   - `final = sum(w_i * member_i(x))` where sum(w_i)=1
   - Can be solved with simple least squares on validation set

8. **Snapshot ensemble** -- Single training run with cyclic LR, save snapshots
   - Train one model with CosineAnnealingWarmRestarts
   - Save checkpoint at each cycle's minimum
   - N snapshots for the cost of ~1 training run
   - Reference: Huang et al., "Snapshot Ensembles: Train 1, Get M for Free" (2017)

9. **MC Dropout** -- Keep dropout active at inference, run N forward passes
   - Zero additional training cost
   - Just set `model.train()` during inference for dropout
   - Average N stochastic forward passes
   - Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016)

### Phase 4: Advanced topics (experiments E10-E12)

10. **CFP ensemble** -- Also ensemble the CFP stage
    - If MLP ensemble reduces variance, CFP ensemble may further smooth output
    - Risk: CFP is already a refinement stage; ensembling may have diminishing returns

11. **Knowledge distillation** -- Train a student from the ensemble teacher
    - Train student MLP with loss = alpha*MSE(student,ground_truth) + (1-alpha)*MSE(student,teacher_output)
    - Student can be smaller: [16->256->256->900] or [16->128->128->900]
    - Measures how much ensemble knowledge compresses into a single model
    - Important for FPGA deployment path

12. **Data augmentation** -- Add noise to RSS during training
    - Augment by adding Gaussian noise to RSS inputs
    - Each member sees slightly different augmented views
    - Orthogonal to other diversity methods; can be combined

## Key principles for ensemble experiments

### What typically helps:
- More diversity between members (different minima = better averaging)
- Moderate ensemble size (3-5 is usually the sweet spot for cost/benefit)
- Combining diversity methods (bagging + seed diversity > seed diversity alone)

### What typically hurts:
- Too many members without diversity (diminishing returns quickly)
- Members so different that some are significantly worse (weak members drag down average)
- Overly complex aggregation methods (weighted averages rarely beat simple mean by much)

### Understanding ensemble + CFP interaction:
- MLP ensemble produces smoother, more stable SLF estimates
- CFP then refines these estimates -- it may need fewer corrections
- If ensemble already reduces artifacts, CFP architecture could potentially be simplified
- A worse ensemble with more consistent errors may lead to better ensemble+CFP

## Metric interpretation

- `test_rmse ~ 0.050`: Excellent single-stage reconstruction
- `cfp_test_rmse ~ 0.050`: Excellent two-stage reconstruction
- `test_ssim ~ 0.80`: Good structural similarity
- `cfp_test_ssim ~ 0.80`: Good post-CFP structural similarity
- Pre-ensemble best: cfp_test_rmse=0.0513, cfp_test_ssim=0.794

The CFP metrics should always be better than MLP-only metrics. If not, the CFP is hurting -- investigate.

## Training time estimates

Single MLP training: ~T minutes
Ensemble N=3: ~3T minutes
Ensemble N=5: ~5T minutes
CFP training: ~T_cfp minutes (independent of ensemble size)
Total: ~N*T + T_cfp must fit within 60 minutes

If T is large, consider: reducing epochs, reducing ensemble size, or using snapshot ensemble.
