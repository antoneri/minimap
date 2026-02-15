#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict minimap parity gate with fixed thresholds and seeds."
    )
    parser.add_argument(
        "--out-dir",
        default=f"/tmp/minimap_parity_gate_{int(time.time())}",
        help="Output directory for parity and benchmark artifacts.",
    )
    parser.add_argument(
        "--minimap",
        default="/Users/anton/kod/minimap/target/release/minimap",
        help="Path to minimap binary.",
    )
    parser.add_argument(
        "--infomap",
        default="/Users/anton/kod/infomap/Infomap",
        help="Path to Infomap binary.",
    )
    parser.add_argument(
        "--parity-seeds",
        default="123,777,2024",
        help="Comma-separated parity seeds.",
    )
    parser.add_argument(
        "--perf-reps",
        type=int,
        default=1,
        help="Performance repetitions for 100k rows in the shared harness.",
    )
    parser.add_argument(
        "--parity-threshold-two-level",
        type=float,
        default=1e-9,
        help="Allowed max absolute codelength delta for two-level parity checks.",
    )
    parser.add_argument(
        "--parity-threshold-multilevel",
        type=float,
        default=2e-3,
        help="Allowed max absolute codelength delta for multilevel parity checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        "scripts/run_infomap_minimap_bench.py",
        "--out-dir",
        str(out_dir),
        "--minimap",
        args.minimap,
        "--infomap",
        args.infomap,
        "--perf-reps",
        str(args.perf_reps),
        "--parity-seeds",
        args.parity_seeds,
        "--parity-threshold-two-level",
        str(args.parity_threshold_two_level),
        "--parity-threshold-multilevel",
        str(args.parity_threshold_multilevel),
        "--strict-parity",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print(f"Parity gate artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
