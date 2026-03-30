---
name: experiment-loop
description: Run the autonomous UWB RTI experiment loop. Use this skill whenever starting a new research session, when the user says "start experiments", "run autoresearch", "kick off experiments", "optimize models", "improve RMSE", or any request to autonomously experiment with the MLP or CFP models. Also trigger when the user says "continue" or "keep going" after a previous session.
---

# UWB RTI Experiment Loop

Run autonomous experiments to minimize test_rmse (and maximize test_ssim) for the UWB Radio Tomographic Imaging system. Each experiment modifies model architecture or training hyperparameters, trains MLP+CFP, evaluates, and keeps or discards.

## Setup (first time only)

1. Check if data exists: `ls data/mlp_data.npz`
2. If not, generate: `python -m uwb_rti.data_generator`
3. Create experiment branch: `git checkout -b autoresearch/<tag>` where tag is today's date
4. Run baseline (unmodified code) as experiment #0
5. Record baseline metrics in `experiment-state.json`

## Experiment loop (runs forever)

### Step 1: Choose next experiment

Read `experiment-state.json`. Check:
- `idea_queue` for untried ideas (pop from front)
- `experiments` history for patterns (what directions improved, what failed)
- If queue is empty, generate new ideas based on what worked

Prioritize experiments that:
- Build on previous improvements (combine two things that each helped)
- Explore dimensions not yet tried
- Are simple and low-risk first, complex later

### Step 2: Implement the change

Modify ONLY the editable files:
- `uwb_rti/models/mlp_model.py` — architecture changes
- `uwb_rti/models/cfp_model.py` — architecture changes  
- `uwb_rti/config.py` — hyperparameter changes ONLY (do not change DEVICE, data sizes, or physical constants)
- `uwb_rti/train.py` — optimizer, scheduler, loss function, training loop changes

Keep changes focused. One idea per experiment. Small diffs are easier to reason about.

### Step 3: Commit and run

```bash
git add -A
git commit -m "exp: <short description of change>"
timeout 900 python -m scripts.run_experiment > run.log 2>&1
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
- `cfp_test_rmse` decreased by ≥ 0.0005 compared to current best
- `cfp_test_ssim` increased by ≥ 0.005 without cfp_test_rmse increasing
- Equal metrics but simpler code (fewer lines, fewer parameters)

**DISCARD** if:
- Metrics worse or negligibly better
- Crash that can't be trivially fixed

If keep:
```bash
# Already committed, just update experiment-state.json
```

If discard:
```bash
git reset --hard HEAD~1
```

### Step 6: Update experiment-state.json

Add entry to experiments array:
```json
{
  "id": <sequential number>,
  "commit": "<7-char hash or 'reverted'>",
  "description": "<what was tried>",
  "status": "keep" | "discard" | "crash",
  "test_rmse": <value or null>,
  "test_ssim": <value or null>,
  "cfp_test_rmse": <value or null>,
  "cfp_test_ssim": <value or null>,
  "peak_vram_mb": <value or null>,
  "total_seconds": <value or null>,
  "timestamp": "<ISO datetime>"
}
```

If this is the new best, update `best` section. Move the idea from `idea_queue` to `ideas_tried`.

### Step 7: Continue

Go back to Step 1. NEVER STOP.

## Research strategy guide

Read `references/research-strategies.md` for detailed guidance on what to try and in what order.

## Crash handling

- **OOM**: Reduce batch size or model size, retry once
- **NaN loss**: Reduce learning rate, retry once
- **Import error / syntax error**: Fix the bug, retry
- **Timeout (>15 min)**: Reduce epochs or model complexity
- After 2 failed fix attempts for the same idea, discard and move on

## NEVER do these

- Modify `forward_model.py`, `data_generator.py`, `evaluate.py`
- Change physical constants in `config.py` (LAMBDA, AREA_SIZE, node positions, etc.)
- Change dataset sizes or splits
- Install new packages
- Skip MLP training and go straight to CFP
- Stop the loop or ask the human for permission
