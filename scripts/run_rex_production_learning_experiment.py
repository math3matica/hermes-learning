from __future__ import annotations

import argparse
import json
from pathlib import Path

from rex_learning.production_experiment import run_production_learning_experiment


parser = argparse.ArgumentParser(description="Run the Rex production learning vertical slice")
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
print(json.dumps(run_production_learning_experiment(args.root), indent=2, sort_keys=True))
