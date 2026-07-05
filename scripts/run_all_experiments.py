"""
Run all probing experiments.

This is the master script that runs everything in sequence:
1. Extract features for each model (if not cached)
2. Run all property probes on each model
3. Generate figures

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --models wavlm_large hubert_large
    python scripts/run_all_experiments.py --properties speaker pitch
"""

import sys
import argparse
import subprocess

sys.path.insert(0, ".")


def run_command(cmd: list):
    """Run a command and handle errors."""
    print(f"\n{'─'*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["wavlm_large", "hubert_large", "whisper_large_v3"],
    )
    parser.add_argument(
        "--properties", nargs="+",
        default=["speaker", "pitch"],
    )
    parser.add_argument(
        "--probes", nargs="+",
        default=["linear"],
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("RUNNING ALL PROBING EXPERIMENTS")
    print("=" * 60)
    print(f"  Models:     {args.models}")
    print(f"  Properties: {args.properties}")
    print(f"  Probes:     {args.probes}")
    
    # Run each combination
    for model in args.models:
        for prop in args.properties:
            for probe in args.probes:
                run_command([
                    sys.executable,
                    "scripts/run_probing.py",
                    "--model", model,
                    "--property", prop,
                    "--probe", probe,
                ])
    
    # Generate figures
    run_command([sys.executable, "scripts/generate_figures.py"])
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
