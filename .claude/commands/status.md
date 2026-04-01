---
name: status
description: Show current ensemble experiment status, best metrics, and recent history
argument-hint: []
---

# /status Command

Display a summary of the ensemble autoresearch session.

## Steps

1. Read `experiment-state.json`
2. Display:
   - Total experiments run
   - Current ensemble configuration (size, diversity method)
   - Current best metrics (test_rmse, test_ssim, cfp versions)
   - Comparison with pre-ensemble baseline
   - Last 5 experiments with status (keep/discard/crash)
   - Number of ideas remaining in idea_queue
   - Current branch and latest commit
3. Run `git log --oneline -5` for recent commits
4. Show current `MLP_ENSEMBLE_SIZE` from config.py

## Usage examples

```
/status
```
