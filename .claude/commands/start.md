---
name: start
description: Start or resume the autonomous experiment loop
argument-hint: [optional: experiment tag like "mar30"]
---

# /start Command

Begin or resume the autonomous UWB RTI experiment loop.

## Steps

1. Read `experiment-state.json` to check current status
2. If status is "ready" (fresh start):
   - Create branch `autoresearch/<tag>` (use argument or today's date)
   - Check if data exists (`ls data/mlp_data.npz`), generate if needed
   - Run baseline experiment (unmodified code)
   - Update experiment-state.json with baseline results
3. If status is "running" (resuming):
   - Run `git log --oneline -10` to see recent work
   - Read experiment history
   - Continue from where we left off
4. Enter the experiment loop (see skill: experiment-loop)

## Usage examples

```
/start mar30
/start
```

## Notes

- First run always establishes baseline
- After baseline, the agent runs autonomously
- Use this command to resume after a session break
