# UWB RTI Autoresearch — Ensemble Edition

## Purpose

Autonomous AI researcher that iteratively improves **ensemble MLP** and CFP deep learning models for UWB Radio Tomographic Imaging. The system uses multiple MLP members with different random seeds, averages their predictions via `EnsembleMLPModel`, then refines the result with CFP. Inspired by Karpathy's autoresearch: modify code, train, evaluate, keep or discard, repeat.

## Architecture overview

```
RSS (16-dim) → [MLP_1, MLP_2, ..., MLP_N] → mean → SLF_rough (30×30) → CFP → SLF_final (30×30)
```

- Each MLP member has the same architecture but different weight init and data shuffle order
- `EnsembleMLPModel` wraps members in `nn.ModuleList` and averages `forward()` outputs
- The ensemble is transparent to `run_experiment.py` and `evaluate.py` (adapter pattern)

## Core workflow

1. Read `experiment-state.json` for current state, best metrics, and experiment history.
2. Pick next experiment idea based on history (what worked, what failed, what's untried).
3. Modify model/training code in `uwb_rti/` — only files listed in "Editable files" below.
4. Git commit the change with a descriptive message.
5. Run the experiment: `python -m scripts.run_experiment > run.log 2>&1`
6. Extract metrics from `run.log` and record in `experiment-state.json`.
7. If improved: keep the commit (advance branch). If worse: `git revert --no-edit HEAD` (NEVER use `git reset --hard` — it destroys uncommitted work).
8. Repeat from step 1. NEVER stop to ask the human.

## Editable files (agent modifies these)

- `uwb_rti/models/mlp_model.py` — MLP and EnsembleMLPModel architecture
- `uwb_rti/models/cfp_model.py` — CFP CNN architecture
- `uwb_rti/config.py` — hyperparameters (including `MLP_ENSEMBLE_SIZE`)
- `uwb_rti/train.py` — training loop, optimizer, scheduler, loss, ensemble training logic
- `uwb_rti/data_generator.py` — data generation pipeline (for augmentation experiments). **When modifying this file, you MUST regenerate data and re-run the baseline to establish a new comparison point. Record the data-generator commit hash in `experiment-state.json` under `data_provenance`.**

## Read-only files (NEVER modify)

- `uwb_rti/forward_model.py` — physics model
- `uwb_rti/evaluate.py` — evaluation metrics (RMSE, SSIM)
- `uwb_rti/visualize.py` — plotting utilities
- `scripts/run_experiment.py` — experiment runner (fixed harness)

## Rules

- The primary metric is `cfp_test_rmse` (lower is better). Secondary: `cfp_test_ssim` (higher is better).
- An experiment "improves" if cfp_test_rmse decreases by ≥ 0.0005 OR cfp_test_ssim increases by ≥ 0.005 without cfp_test_rmse increasing.
- Time budget: 60 minutes per experiment. Ensemble training takes N× single-model time — plan accordingly.
- If `MLP_ENSEMBLE_SIZE > 3`, verify training fits within budget before committing.
- If a run crashes (OOM, bug), attempt a quick fix. If unfixable after 2 attempts, discard and revert.
- Do NOT install new packages. Only use what's in `requirements.txt`.
- Do NOT modify evaluation — that is the ground truth.
- Always train MLP ensemble first, then generate CFP pairs, then train CFP. Never skip stages.
- `train_mlp()` must return `(ensemble_model, history)` — the ensemble model must be an `nn.Module` with a standard `forward()`.
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

# Run a single experiment (trains ensemble MLP + CFP, evaluates)
python -m scripts.run_experiment > run.log 2>&1

# Check results
grep "test_rmse\|test_ssim\|cfp_test" run.log

# Check for crash
tail -n 30 run.log
```
