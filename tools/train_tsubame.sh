#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs/tsubame

if [[ $# -ne 0 ]]; then
    echo "Usage: tools/train_tsubame.sh" >&2
    exit 2
fi

# Stop five minutes before the allocation ends to leave time for the exit save.
allocation_deadline=${BUCKSHOT_ALLOCATION_DEADLINE:?Set the allocated job deadline in Unix seconds}
.venv/bin/python - "$allocation_deadline" <<'PY'
import json
from pathlib import Path
import sys
import time

configuration = json.loads(Path('configs/tsubame.json').read_text())
remaining_seconds = int(sys.argv[1]) - time.time() - 300
if remaining_seconds < 60:
    raise RuntimeError('Allocation has insufficient time for a safe training launch')
configuration['training']['max_runtime_seconds'] = remaining_seconds
Path('runs/tsubame/launch_config.json').write_text(json.dumps(configuration, indent=2))
print('TRAINING_RUNTIME_SECONDS', remaining_seconds, file=sys.stderr, flush=True)
PY

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

.venv/bin/python -m roulette train --config runs/tsubame/launch_config.json --resume