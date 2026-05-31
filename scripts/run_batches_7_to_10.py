"""Chain runner: execute H432 through H448 sequentially.

H431 is already running in parallel; this script runs H432-H448.
Run after H431 completes if you want to avoid conflicts.
Otherwise, run this immediately for parallel execution.

Run: uv run python scripts/run_batches_7_to_10.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "run_h432.py", "run_h433.py", "run_h434.py", "run_h435.py",
    "run_h436.py", "run_h437.py", "run_h438.py", "run_h439.py", "run_h440.py",
    "run_h441.py", "run_h442.py", "run_h443.py", "run_h444.py", "run_h445.py",
    "run_h446.py", "run_h447.py", "run_h448.py",
]

scripts_dir = Path(__file__).parent
for script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print('='*60)
    result = subprocess.run(
        [sys.executable, str(scripts_dir / script)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"WARNING: {script} exited with code {result.returncode}")

print("\nAll H432-H448 complete.")
