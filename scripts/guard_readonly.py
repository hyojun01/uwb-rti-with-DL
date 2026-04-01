"""
Pre-tool-use hook: blocks writes to read-only files.
Exit 0 = allow, exit 1 = block.
"""
import sys
import os
import json

READONLY_FILES = {
    "uwb_rti/forward_model.py",
    "uwb_rti/evaluate.py",
    "uwb_rti/visualize.py",
    "uwb_rti/main.py",
    "uwb_rti/validate_model.py",
    "scripts/run_experiment.py",
    "scripts/guard_readonly.py",
}

# Read the tool input from stdin (Claude Code passes JSON)
try:
    input_data = json.loads(sys.stdin.read())
    file_path = input_data.get("file_path", "") or input_data.get("path", "")
    # Normalize path
    file_path = os.path.relpath(file_path) if file_path else ""

    if file_path in READONLY_FILES:
        print(f"BLOCKED: {file_path} is read-only. Do not modify evaluation, data generation, or experiment runner files.", file=sys.stderr)
        sys.exit(1)
except Exception:
    pass

sys.exit(0)
