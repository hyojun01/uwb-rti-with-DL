---
name: status
description: Show current experiment status, best metrics, and recent history
argument-hint: []
---

# /status Command

Display a summary of the autoresearch session.

## Steps

1. Read `experiment-state.json`
2. Display:
   - Total experiments run
   - Current best metrics (test_rmse, test_ssim, cfp versions)
   - Last 5 experiments with status (keep/discard/crash)
   - Number of ideas remaining in queue
   - Current branch and latest commit
3. Run `git log --oneline -5` for recent commits

## Usage examples

```
/status
```
