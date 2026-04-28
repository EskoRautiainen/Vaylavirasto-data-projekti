from pathlib import Path
import subprocess
import sys
import time
import os

#NOTE: check single lane road lane seperation per direction, and if it is possible to split them into separate features

# --------------------------------------------------
# SETTINGS AND TOGGLES
# --------------------------------------------------
RUN_DATA_HANDLER = True
RUN_DATA_TO_VISUAL = False
RUN_PERFORMANCE_BENCHMARK = False
# RUN_NOTE will be added to the benchmark log to help identify the context of this run.
RUN_NOTE = "Digiroad only relevant columns"  # Example: "Test run with latest code changes, including lane index calculation refactor and enhanced performance logging."

# Optional: stop immediately if one script fails
STOP_ON_ERROR = True

# Optional: show timing for each script
SHOW_TIMINGS = True
# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
try:
    codes_dir = Path(__file__).resolve().parent
except NameError:
    codes_dir = Path.cwd()

python_executable = sys.executable

scripts = [
    ("DataHandler.ipynb", RUN_DATA_HANDLER),
    ("DataToVisual.ipynb", RUN_DATA_TO_VISUAL),
    ("PerformanceBenchmark.py", RUN_PERFORMANCE_BENCHMARK),
]


def run_script(script_name):
    script_path = codes_dir / script_name

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    print(f"\n--- Running {script_name} ---")

    start_time = time.perf_counter()

    import os
    env = os.environ.copy()
    env["RUN_NOTE"] = RUN_NOTE

    # Decide how to run based on file type
    if script_path.suffix == ".ipynb":
        cmd = [
            python_executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(script_path)
        ]
    else:
        cmd = [python_executable, str(script_path)]

    result = subprocess.run(
        cmd,
        cwd=str(codes_dir),
        env=env
    )

    elapsed = time.perf_counter() - start_time

    if SHOW_TIMINGS:
        print(f"Finished {script_name} in {elapsed:.2f} s")

    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed with exit code {result.returncode}")
        return False

    print(f"[OK] {script_name} completed successfully")
    return True


def main():
    enabled_scripts = [name for name, enabled in scripts if enabled]

    if not enabled_scripts:
        print("No scripts are enabled. Set at least one RUN_* toggle to True.")
        return

    print("RunAll starting...")
    print("Enabled scripts:")
    for name in enabled_scripts:
        print(f" - {name}")

    total_start = time.perf_counter()

    for script_name, enabled in scripts:
        if not enabled:
            print(f"\n--- Skipping {script_name} ---")
            continue

        success = run_script(script_name)

        if not success and STOP_ON_ERROR:
            print("\nRunAll stopped because a script failed.")
            return

    total_elapsed = time.perf_counter() - total_start
    print(f"\nRunAll finished in {total_elapsed:.2f} s")


if __name__ == "__main__":
    main()