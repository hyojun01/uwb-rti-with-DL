---
name: experiment-loop
description: Run the autonomous UWB RTI ensemble experiment loop. Use this skill whenever starting a new research session, when the user says "start experiments", "run autoresearch", "kick off experiments", "optimize models", "improve RMSE", or any request to autonomously experiment with the ensemble MLP or CFP models. Also trigger when the user says "continue" or "keep going" after a previous session.
---

# UWB RTI Ensemble Experiment Loop

Run autonomous experiments to minimize cfp_test_rmse (and maximize cfp_test_ssim) for the UWB Radio Tomographic Imaging system. The MLP stage uses an ensemble of multiple models with averaged predictions. Each experiment modifies ensemble configuration, model architecture, or training hyperparameters, trains the full pipeline, evaluates, and keeps or discards.

## Setup (first time only)

1. Check if data exists: `ls data/mlp_data.npz`
2. If not, generate: `python -m uwb_rti.data_generator`
3. Create experiment branch: `git checkout -b autoresearch/<tag>` where tag is today's date
4. Run baseline (current ensemble code, unmodified) as experiment #0
5. Record baseline metrics in `experiment-state.json`

## Experiment loop (runs forever)

### Step 1: Choose next experiment

Read `experiment-state.json`. Check:
- `idea_queue` for untried ideas (pop from front)
- `experiments` history for patterns (what directions improved, what failed)
- If queue is empty, generate new ideas based on what worked
- Consider ensemble-specific ideas: size, diversity method, aggregation, snapshot ensemble, KD

Prioritize experiments that:
- Build on previous improvements (combine two things that each helped)
- Explore dimensions not yet tried
- Are simple and low-risk first, complex later
- Fit within the 60-minute time budget (ensemble size x single training time)

### Step 2: Implement the change

Modify ONLY the editable files:
- `uwb_rti/models/mlp_model.py` -- MLP and EnsembleMLPModel architecture
- `uwb_rti/models/cfp_model.py` -- architecture changes
- `uwb_rti/config.py` -- hyperparameters including `MLP_ENSEMBLE_SIZE`
- `uwb_rti/train.py` -- optimizer, scheduler, loss, ensemble training logic
- `uwb_rti/data_generator.py` -- data augmentation (preserve output format). **If modified: regenerate data, re-run baseline, update `data_provenance` in experiment-state.json with the new commit hash.**

Keep changes focused. One idea per experiment. Small diffs are easier to reason about.

**Ensemble-specific considerations:**
- When changing ensemble size, verify total training time fits within budget
- When adding member diversity (bagging, HP variation), keep one change at a time
- When modifying EnsembleMLPModel, ensure `forward()` still returns (batch, 900)
- The returned model from `train_mlp()` must be compatible with `evaluate.py`

### Step 3: Commit and run

```bash
git add -A
git commit -m "exp: <short description of change>"
timeout 3600 python -m scripts.run_experiment > run.log 2>&1
```

### Step 4: Read results

```bash
grep "^test_rmse\|^test_ssim\|^cfp_test_rmse\|^cfp_test_ssim\|^status\|^peak_vram_mb\|^total_seconds" run.log
```

If grep returns nothing, the run crashed:
```bash
tail -n 50 run.log
```

### Step 5: Decide keep or discard

**KEEP** if any of:
- `cfp_test_rmse` decreased by >= 0.0005 compared to current best
- `cfp_test_ssim` increased by >= 0.005 without cfp_test_rmse increasing
- Equal metrics but fewer total parameters or faster inference

**DISCARD** if:
- Metrics worse or negligibly better
- Crash that can't be trivially fixed

If keep: already committed, just update experiment-state.json

If discard:
```bash
# Safe rollback — only reverts the experiment commit, preserves other uncommitted work
git revert --no-edit HEAD
```
If `git revert` fails (e.g., conflicts with uncommitted changes), restore only the experiment-scoped files:
```bash
git show HEAD~1:uwb_rti/models/mlp_model.py > uwb_rti/models/mlp_model.py
git show HEAD~1:uwb_rti/models/cfp_model.py > uwb_rti/models/cfp_model.py
git show HEAD~1:uwb_rti/config.py > uwb_rti/config.py
git show HEAD~1:uwb_rti/train.py > uwb_rti/train.py
git show HEAD~1:uwb_rti/data_generator.py > uwb_rti/data_generator.py
git commit -am "revert: <experiment description>"
```
NEVER use `git reset --hard` — it destroys unrelated uncommitted work.

### Step 6: Update experiment-state.json

Add entry to experiments array. All metric fields MUST be numeric (float) or null — NEVER strings:
```json
{
  "id": 1,
  "commit": "abc1234",
  "description": "what was tried",
  "status": "keep",
  "test_rmse": 0.0521,
  "test_ssim": 0.774,
  "cfp_test_rmse": 0.0512,
  "cfp_test_ssim": 0.793,
  "ensemble_size": 3,
  "peak_vram_mb": 1024.5,
  "total_seconds": 845.2,
  "timestamp": "2026-04-01T12:00:00Z"
}
```
Type rules:
- `id`: integer (sequential, starting from 1)
- `commit`: string (7-char git hash, or `"reverted"`)
- `description`: string
- `status`: string enum — one of `"keep"`, `"discard"`, `"crash"`
- `test_rmse`, `test_ssim`, `cfp_test_rmse`, `cfp_test_ssim`: number or null
- `ensemble_size`: integer
- `peak_vram_mb`, `total_seconds`: number or null
- `timestamp`: string (ISO 8601)

Also update `ensemble_config` if ensemble parameters changed.
If this is the new best, update `best` section. Move the idea from `idea_queue` to `ideas_tried`.

### Step 7: Continue

Go back to Step 1. NEVER STOP.

## Research strategy guide

Read `references/research-strategies.md` for detailed guidance on what to try and in what order.

## Crash handling

- **OOM**: Reduce ensemble size or batch size, retry once
- **NaN loss**: Reduce learning rate, retry once
- **Import error / syntax error**: Fix the bug, retry
- **Timeout (>60 min)**: Reduce ensemble size, epochs, or model complexity
- After 2 failed fix attempts for the same idea, discard and move on

## NEVER do these

- Modify `forward_model.py`, `evaluate.py`, `run_experiment.py`
- Change physical constants in `config.py` (LAMBDA, AREA_SIZE, node positions, etc.)
- Change dataset sizes or splits
- Install new packages
- Skip MLP training and go straight to CFP
- Return something other than `(nn.Module, history_dict)` from `train_mlp()`
- Stop the loop or ask the human for permission
