from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple learning-rate sweep")
    parser.add_argument("--config", default=None, help="Optional base config overrides")
    parser.add_argument(
        "--lrs",
        default="1e-5,2e-5,3e-5,5e-5",
        help="Comma-separated list of learning rates to try",
    )
    parser.add_argument("--output-root", default="outputs/lr_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lrs = [lr.strip() for lr in args.lrs.split(",") if lr.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for lr in lrs:
        output_dir = output_root / f"lr_{lr}"
        cmd = [
            "numina-finetune",
            "--learning-rate",
            lr,
            "--output-dir",
            str(output_dir),
        ]
        if args.config:
            cmd.extend(args.config.split())
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
