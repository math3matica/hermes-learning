from __future__ import annotations

import argparse
import json
from pathlib import Path

from rex_learning.behavioral_experiment import run_behavioral_reuse_experiment


parser = argparse.ArgumentParser(description="Run Rex Learning behavioral reuse qualification")
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
print(json.dumps(run_behavioral_reuse_experiment(args.root), indent=2, sort_keys=True))
