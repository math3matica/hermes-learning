from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rex_learning.experiment import run_bootstrap_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Rex Learning V1 bootstrap experiment")
    parser.add_argument("--root", type=Path, required=True, help="empty durable learning artifact root")
    args = parser.parse_args()
    print(json.dumps(run_bootstrap_experiment(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
