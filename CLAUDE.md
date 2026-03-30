# UWB RTI Autoresearch

## Purpose

Autonomous AI researcher that iteratively improves MLP and CFP deep learning models for UWB Radio Tomographic Imaging. Inspired by Karpathy's autoresearch: modify code, train, evaluate, keep or discard, repeat.

## Core workflow

1. Read `experiment-state.json` for current state, best metrics, and experiment history.
2. Pick next experiment idea based on history (what worked, what failed, what's untried).
3. Modify model/training code in `uwb_rti/` — only files listed in "Editable files" below.
4. Git commit the change with a descriptive message.
5. Run the experiment: `python -m scripts.run_experiment > run.log 2>&1`
6. Extract metrics from `run.log` and record in `experiment-state.json`.
7. If improved: keep the commit (advance branch). If worse: `git reset --hard HEAD~1`.
8. Repeat from step 1. NEVER stop to ask the human.

## Editable files (agent modifies these)

- `uwb_rti/models/mlp_model.py` — MLP architecture
- `uwb_rti/models/cfp_model.py` — CFP CNN architecture
- `uwb_rti/config.py` — hyperparameters only (learning rates, batch size, epochs, dropout, model dimensions)
- `uwb_rti/train.py` — training loop, optimizer, scheduler, loss function

## Read-only files (NEVER modify)

- `uwb_rti/forward_model.py` — physics model
- `uwb_rti/data_generator.py` — data generation pipeline
- `uwb_rti/evaluate.py` — evaluation metrics (RMSE, SSIM)
- `uwb_rti/visualize.py` — plotting utilities
- `scripts/run_experiment.py` — experiment runner (fixed harness)

## Rules

- The primary metric is `test_rmse` (lower is better). Secondary: `test_ssim` (higher is better).
- An experiment "improves" if test_rmse decreases by at least 0.0005 OR test_ssim increases by at least 0.005 without test_rmse increasing.
- Each experiment has a fixed time budget. If training exceeds 15 minutes, kill and treat as crash.
- If a run crashes (OOM, bug), attempt a quick fix. If unfixable after 2 attempts, discard and revert.
- Keep the simplicity criterion: small improvement + ugly complexity = not worth it. Equal performance + simpler code = keep.
- Do NOT install new packages. Only use what's in `requirements.txt`.
- Do NOT modify evaluation or data generation — those are the ground truth.
- Always train MLP first, then generate CFP pairs, then train CFP. Never skip stages.
- Log every experiment to `experiment-state.json` — even crashes.

## Context management

- Read `experiment-state.json` at session start.
- Update it after every experiment.
- Run `git log --oneline -20` to understand recent work.
- This is a long-running task. Do not stop or ask for confirmation.

## Quick reference

```bash
# Generate data (one-time, if data/ doesn't exist)
python -m uwb_rti.data_generator

# Run a single experiment (trains MLP + CFP, evaluates)
python -m scripts.run_experiment > run.log 2>&1

# Check results
grep "test_rmse\|test_ssim" run.log

# Check for crash
tail -n 30 run.log
```
