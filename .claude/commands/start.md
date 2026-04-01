---
name: start
description: Start or resume the autonomous ensemble experiment loop
argument-hint: [optional: experiment tag like "apr01"]
---

# /start Command

Begin or resume the autonomous UWB RTI ensemble experiment loop.

## Steps

1. Read `experiment-state.json` to check current status
2. If status is "ready" (fresh start):
   - Create branch `autoresearch/<tag>` (use argument or today's date)
   - Check if data exists (`ls data/mlp_data.npz`), generate if needed
   - Run baseline experiment (current ensemble code, unmodified)
   - Update experiment-state.json with baseline results and `ensemble_config`
3. If status is "running" (resuming):
   - Run `git log --oneline -10` to see recent work
   - Read experiment history
   - Check current `MLP_ENSEMBLE_SIZE` and ensemble configuration
   - Continue from where we left off
4. Enter the experiment loop (see skill: experiment-loop)

## Usage examples

```
/start apr01
/start
```

## Notes

- First run always establishes ensemble baseline (N=3, seed-only diversity)
- After baseline, the agent runs autonomously through the idea queue
- The idea queue is pre-populated with ensemble-specific experiments
- Use this command to resume after a session break
