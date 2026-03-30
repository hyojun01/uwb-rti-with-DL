# UWB RTI Autoresearch

Autonomous AI researcher for UWB Radio Tomographic Imaging model optimization. Adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) principles to the UWB RTI domain.

## Concept

An AI agent autonomously experiments with MLP and CFP model architectures and training hyperparameters:

1. **Modify** — Change model architecture or training config
2. **Train** — Run full MLP → CFP pipeline
3. **Evaluate** — Measure test_rmse and test_ssim
4. **Keep or discard** — Better results advance the branch; worse results get reverted
5. **Repeat** — The agent never stops until manually interrupted

## Quick start

```bash
# 1. Copy UWB RTI source files into uwb_rti/
# 2. Install dependencies
pip install torch numpy matplotlib scikit-image

# 3. Generate data (one-time)
python -m uwb_rti.data_generator

# 4. Start Claude Code
cd uwb-rti-autoresearch && claude

# 5. Kick off experiments
> /start mar30
```

## Structure

| Component | Location | Role |
|-----------|----------|------|
| CLAUDE.md | `./CLAUDE.md` | Core rules (loaded every session) |
| Experiment runner | `scripts/run_experiment.py` | Fixed training+eval pipeline |
| Guard hook | `scripts/guard_readonly.py` | Blocks writes to read-only files |
| Experiment state | `experiment-state.json` | Progress tracking (JSON) |
| Skill | `.claude/skills/experiment-loop/` | Detailed experiment loop instructions |
| Research guide | `.claude/skills/.../references/` | What to try and in what order |
| Analyzer agent | `.claude/agents/analyzer.md` | Suggests next experiments |
| Commands | `.claude/commands/` | `/start`, `/status` entry points |
| Rules | `.claude/rules/` | File-specific constraints |

## Architecture

```
Human ──(edits CLAUDE.md / research-strategies.md)──→ Harness
                                                        │
AI Agent ←──(reads experiment-state.json)───────────────┘
    │
    ├── Modifies: mlp_model.py, cfp_model.py, config.py, train.py
    │
    ├── Runs: scripts/run_experiment.py (MLP train → CFP train → Eval)
    │
    ├── Evaluates: test_rmse (↓ better), test_ssim (↑ better)
    │
    ├── keep → git commit stays, branch advances
    │   discard → git reset --hard HEAD~1
    │
    └── Loops forever (≈12 experiments/hour at 5min each)
```

## Key design decisions

1. **Fixed evaluation**: `evaluate.py` and `forward_model.py` are read-only so the agent can't game metrics
2. **Separate data splits**: MLP and CFP use different datasets to prevent overfitting
3. **Time budget**: 15 min max per experiment (kills runaways)
4. **Idea queue**: Pre-seeded with ~30 experiment ideas, agent generates more as needed
5. **JSON state**: Models treat JSON as code → more careful modifications than Markdown

## Adapting autoresearch for UWB RTI

| autoresearch | UWB RTI autoresearch |
|---|---|
| Single file (train.py) | Four editable files (2 models + config + train) |
| Single metric (val_bpb) | Two metrics (test_rmse primary, test_ssim secondary) |
| 5 min time budget | 15 min budget (MLP + CFP + eval) |
| prepare.py read-only | forward_model + data_generator + evaluate read-only |
| One model | Two-stage pipeline (MLP → CFP) |
| program.md for human | CLAUDE.md + skills + rules for human |
